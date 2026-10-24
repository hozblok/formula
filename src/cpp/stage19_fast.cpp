#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
using C = std::complex<double>;
constexpr double pi = 3.1415926535897932384626433832795;
using Row = std::array<double,13>;

template<class T> py::bytes packed(const std::vector<T>& a) {
    return py::bytes(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));
}
template<class T> std::vector<T> unpacked(py::buffer buffer) {
    auto b=buffer.request();
    if(b.ndim!=1 || b.strides[0]!=b.itemsize || (b.size*b.itemsize)%sizeof(T))
        throw std::invalid_argument("Expected contiguous binary buffer");
    std::vector<T> a(b.size*b.itemsize/sizeof(T));
    std::memcpy(a.data(),b.ptr,a.size()*sizeof(T)); return a;
}
std::pair<std::vector<double>,std::vector<double>> gauss(int n) {
    if(n<1 || n>4096)throw std::invalid_argument("Gauss order must be in [1,4096]");
    std::vector<double> x(n),w(n);
    for(int i=0;i<(n+1)/2;++i) {
        double z=std::cos(pi*(i+.75)/(n+.5)),pp=0;
        for(int it=0;it<60;++it) {
            double p=1,q=0;
            for(int j=1;j<=n;++j) {double r=q;q=p;p=((2*j-1)*z*q-(j-1)*r)/j;}
            pp=n*(z*p-q)/(z*z-1);
            double delta=p/pp; z-=delta;if(std::abs(delta)<2e-15)break;
        }
        x[i]=(1-z)/2;x[n-1-i]=(1+z)/2;
        w[i]=w[n-1-i]=1/((1-z*z)*pp*pp);
    } return {x,w};
}
py::dict sources(const std::vector<Row>& rows,double kappa,
    double xmin,double xmax,double ymin,double ymax,int minimum,double safety,
    int maximum,size_t point_limit) {
    if(!(kappa>0) || minimum<2 || maximum<minimum || maximum>4096 || safety<=0
       || !std::isfinite(kappa) || !std::isfinite(safety) ||
       !std::isfinite(xmin+xmax+ymin+ymax) || xmin>xmax || ymin>ymax)
        throw std::invalid_argument("Invalid quadrature options");
    for(const auto&row:rows)for(double v:row)if(!std::isfinite(v))
        throw std::invalid_argument("Nonfinite triangle");
    std::vector<double> xs,ys;std::vector<C> cs;
    std::map<int,std::pair<std::vector<double>,std::vector<double>>> rules;
    int used_min=maximum,used_max=0;
    {
      py::gil_scoped_release unlock;
      for(const auto&r:rows) {
        double e1x=r[2]-r[0],e1y=r[3]-r[1],e2x=r[4]-r[0],e2y=r[5]-r[1];
        double det=std::abs(e1x*e2y-e1y*e2x);
        double b1=0,b2=0;
        for(int a=0;a<3;++a)for(double px:{xmin,xmax})for(double pyv:{ymin,ymax}){
          double gx=r[8]+kappa*(r[2*a]-px),gy=r[9]+kappa*(r[2*a+1]-pyv);
          b1=std::max(b1,std::abs(gx*e1x+gy*e1y));
          b2=std::max(b2,std::abs(gx*e2x+gy*e2y));
        }
        double order_u=minimum+std::ceil(safety*(b1+b2)/3);
        double order_v=minimum+std::ceil(safety*b2/3);
        if(!std::isfinite(order_u+order_v) || order_u>maximum || order_v>maximum)
            throw std::runtime_error("Source quadrature order limit exceeded");
        int nu=int(order_u),nv=int(order_v);
        if(nu>maximum || nv>maximum)throw std::runtime_error("Source quadrature order limit exceeded");
        if(xs.size()+size_t(nu)*nv>point_limit)throw std::runtime_error("Source quadrature point limit exceeded; reduce batch size");
        used_min=std::min({used_min,nu,nv});used_max=std::max({used_max,nu,nv});
        if(!rules.count(nu))rules[nu]=gauss(nu);
        if(!rules.count(nv))rules[nv]=gauss(nv);
        auto &ru=rules[nu],&rv=rules[nv];C amp(r[10],r[11]);
        for(int i=0;i<nu;++i)for(int j=0;j<nv;++j){
          double u=ru.first[i],v=(1-u)*rv.first[j];
          double x=r[0]+u*e1x+v*e2x,y=r[1]+u*e1y+v*e2y;
          double phase=r[12]+r[8]*(x-r[6])+r[9]*(y-r[7])+.5*kappa*(x*x+y*y);
          double weight=det*ru.second[i]*rv.second[j]*(1-u);
          xs.push_back(x);ys.push_back(y);cs.push_back(weight*amp*std::polar(1.,phase));
        }
      }
    }
    py::dict out;out["x"]=packed(xs);out["y"]=packed(ys);out["coefficients"]=packed(cs);
    out["points"]=xs.size();out["min_order"]=used_min;out["max_order"]=used_max;return out;
}
py::dict prepare(py::buffer xb,py::buffer yb,py::buffer cb,
                 double scale_x,double scale_y,double carrier_x,double carrier_y) {
    auto x=unpacked<double>(xb),y=unpacked<double>(yb);auto c=unpacked<C>(cb);
    if(x.size()!=y.size() || x.size()!=c.size())throw std::invalid_argument("Mismatched source buffers");
    {
      py::gil_scoped_release unlock;
      for(size_t i=0;i<x.size();++i){
        c[i]*=std::polar(1.,-carrier_x*x[i]-carrier_y*y[i]);
        x[i]=std::remainder(scale_x*x[i],2*pi);y[i]=std::remainder(scale_y*y[i],2*pi);
      }
    }
    py::dict out;out["x"]=packed(x);out["y"]=packed(y);out["coefficients"]=packed(c);return out;
}
py::bytes strengths(py::buffer xb,py::buffer yb,py::buffer cb,double cx,double cy){
    auto x=unpacked<double>(xb),y=unpacked<double>(yb);auto c=unpacked<C>(cb);
    if(x.size()!=y.size() || x.size()!=c.size())throw std::invalid_argument("Mismatched source buffers");
    {py::gil_scoped_release unlock;for(size_t i=0;i<x.size();++i)c[i]*=std::polar(1.,-cx*x[i]-cy*y[i]);}
    return packed(c);
}
void postprocess(py::buffer spectrum,py::buffer output,
                 const std::vector<double>& x,const std::vector<double>& y,
                 double kappa,double ox,double oy,double weight) {
    auto f=spectrum.request(),o=output.request();
    size_t n=x.size()*y.size();
    if(!n || f.ndim!=1 || o.ndim!=1 || f.strides[0]!=f.itemsize || o.strides[0]!=o.itemsize
       || o.readonly || f.size*f.itemsize!=n*sizeof(C) || o.size*o.itemsize!=n*sizeof(C))
       throw std::invalid_argument("Invalid field buffers");
    auto src=static_cast<const C*>(f.ptr);auto dst=static_cast<C*>(o.ptr);
    py::gil_scoped_release unlock;
    C pref(0,-weight*kappa/(2*pi));
    for(size_t iy=0;iy<y.size();++iy)for(size_t ix=0;ix<x.size();++ix){
      double px=x[ix]+ox,pyv=y[iy]+oy;
      size_t j=iy*x.size()+ix;dst[j]+=pref*std::polar(1.,.5*kappa*(px*px+pyv*pyv))*src[j];
    }
}
}

void bind_stage19_fast(py::module_& m) {
  m.def("quadrature_sources",&sources,py::arg("records"),py::arg("kappa"),
      py::arg("xmin"),py::arg("xmax"),py::arg("ymin"),py::arg("ymax"),
      py::arg("minimum")=6,py::arg("safety")=1.,py::arg("maximum")=512,
      py::arg("point_limit")=10000000);
  m.def("nufft_prepare",&prepare);
  m.def("nufft_strengths",&strengths);
  m.def("nufft_postprocess",&postprocess);
  m.def("gauss_rule",&gauss);
}
