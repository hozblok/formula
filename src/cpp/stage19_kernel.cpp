// Standalone, NumPy-free Stage 19 field integration controls.
// A record is [x0,y0,x1,y1,x2,y2, anchor_x,anchor_y, gx,gy,A_re,A_im,phase].
// E(q) = A exp(i*(phase + gx*(qx-anchor_x) + gy*(qy-anchor_y))).
// Fresnel propagation omits the common exp(i*k*d) phase, explicitly reported.
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;
void bind_stage19_fast(py::module_& m);
namespace {
using C = std::complex<double>;
constexpr double PI = 3.141592653589793238462643383279502884;
const C I(0, 1);
struct V {
    double x, y;
    V operator+(V b) const { return {x+b.x,y+b.y}; }
    V operator-(V b) const { return {x-b.x,y-b.y}; }
    V operator*(double a) const { return {x*a,y*a}; }
};
double dot(V a,V b) { return a.x*b.x+a.y*b.y; }
double cross(V a,V b) { return a.x*b.y-a.y*b.x; }
double norm2(V a) { return dot(a,a); }
bool finite(double a) { return std::isfinite(a); }
C cis(double a) { return {std::cos(a),std::sin(a)}; }
double sinc(double a) {
    if (std::abs(a)<1e-4) { double a2=a*a; return 1-a2/6+a2*a2/120-a2*a2*a2/5040; }
    return std::sin(a)/a;
}
// expm1(i*t)/(i*t), with no subtraction of nearly equal real numbers.
C exprel_i(double t) {
    return cis(t/2)*sinc(t/2);
}
struct Sum {
    C value{}, compensation{};
    void add(C x) { C y=x-compensation, t=value+y; compensation=(t-value)-y; value=t; }
};
struct Triangle {
    std::array<V,3> q;
    V anchor,g,center;
    C a;
    double phase,area,radius;
};
std::vector<Triangle> parse(const std::vector<std::vector<double>>& records) {
    std::vector<Triangle> ts; ts.reserve(records.size());
    for (const auto& r:records) {
        if (r.size()!=13) throw std::invalid_argument("triangle record must contain 13 numbers");
        for(double v:r) if(!finite(v)) throw std::invalid_argument("triangle record must be finite");
        Triangle t{{V{r[0],r[1]},V{r[2],r[3]},V{r[4],r[5]}}, {r[6],r[7]}, {r[8],r[9]}, {}, {r[10],r[11]}, r[12], 0, 0};
        double det=cross(t.q[1]-t.q[0],t.q[2]-t.q[0]);
        if (!finite(det)||det==0) throw std::invalid_argument("degenerate or overflowing exit triangle");
        if(det<0) std::swap(t.q[1],t.q[2]);
        t.area=std::abs(det)/2;
        t.center=t.q[0]+((t.q[1]-t.q[0])+(t.q[2]-t.q[0]))*(1.0/3);
        for(V q:t.q)t.radius=std::max(t.radius,std::sqrt(norm2(q-t.center)));
        ts.push_back(t);
    }
    return ts;
}
void check_axes(const std::vector<double>& x,const std::vector<double>& y) {
    for(const auto* a:{&x,&y}) {
        if(a->empty()) throw std::invalid_argument("axes must be nonempty");
        for(std::size_t n=0;n<a->size();++n)
            if(!finite((*a)[n])||(n&&(*a)[n]<=(*a)[n-1]))
                throw std::invalid_argument("axes must be finite and strictly increasing");
    }
    if(x.size()>std::numeric_limits<std::size_t>::max()/y.size()) throw std::invalid_argument("grid size overflow");
}
struct Rule { std::vector<double> x,w; };
Rule gauss(int n) {
    if(n<1||n>64) throw std::invalid_argument("quadrature order must lie in 1..64");
    Rule r; r.x.resize(n);r.w.resize(n);
    for(int i=0;i<(n+1)/2;++i) {
        double z=std::cos(PI*(i+.75)/(n+.5)),deriv=0;
        for(int iteration=0;iteration<50;++iteration) {
            double p0=1,p1=z;
            for(int j=2;j<=n;++j){double p=((2*j-1)*z*p1-(j-1)*p0)/j;p0=p1;p1=p;}
            double pn=n==1?z:p1,pnm=n==1?1:p0;
            deriv=n*(z*pn-pnm)/(z*z-1);
            double step=pn/deriv;z-=step;
            if(std::abs(step)<4*std::numeric_limits<double>::epsilon())break;
        }
        // Re-evaluate the derivative at the converged root.
        double p0=1,p1=z;
        for(int j=2;j<=n;++j){double p=((2*j-1)*z*p1-(j-1)*p0)/j;p0=p1;p1=p;}
        deriv=n*(z*(n==1?z:p1)-(n==1?1:p0))/(z*z-1);
        r.x[i]=(1-z)/2;r.x[n-1-i]=(1+z)/2;
        r.w[i]=r.w[n-1-i]=1/((1-z*z)*deriv*deriv);
    }
    return r;
}
struct Stats {
    std::uint64_t pairs=0,regularized=0,edge_nodes=0,max_subdivisions=0;
    double absolute_edge_sum=0;
    void add(const Stats& s) {
        pairs+=s.pairs;regularized+=s.regularized;edge_nodes+=s.edge_nodes;
        max_subdivisions=std::max(max_subdivisions,s.max_subdivisions);
        absolute_edge_sum+=s.absolute_edge_sum;
    }
};
double phase_at(const Triangle& t,V q,V p,double kappa) {
    return t.phase+dot(t.g,q-t.anchor)+.5*kappa*norm2(q-p);
}
C triangle_fresnel(const Triangle& t,V p,double kappa,const Rule& rule,
                   double phase_step,std::size_t max_subdivisions,Stats& stats) {
    V q0=p-t.g*(1/kappa);
    bool regular=norm2(q0-t.center)<=16*t.radius*t.radius;
    C phase0=regular?cis(phase_at(t,q0,p,kappa)):C{};
    Sum sum;
    ++stats.pairs;if(regular)++stats.regularized;
    for(int e=0;e<3;++e) {
        V start=t.q[e],edge=t.q[(e+1)%3]-start,normal{edge.y,-edge.x};
        double der0=dot(t.g+(start-p)*kappa,edge),der1=der0+kappa*norm2(edge);
        double wanted=std::ceil(std::max(std::abs(der0),std::abs(der1))/phase_step);
        if(!finite(wanted)||wanted>static_cast<double>(max_subdivisions))
            throw std::runtime_error("contour edge exceeds max_subdivisions; increase limit or refine geometry");
        std::size_t count=std::max<std::size_t>(1,static_cast<std::size_t>(wanted));
        stats.max_subdivisions=std::max(stats.max_subdivisions,static_cast<std::uint64_t>(count));
        for(std::size_t s=0;s<count;++s) for(std::size_t j=0;j<rule.x.size();++j) {
            V q=start+edge*((s+rule.x[j])/count),r=q-q0;
            double r2=norm2(r),rn=dot(r,normal);
            C v=regular?.5*rn*exprel_i(.5*kappa*r2)*phase0
                       :cis(phase_at(t,q,p,kappa))*rn/(I*kappa*r2);
            v*=rule.w[j]/count;
            sum.add(v);stats.absolute_edge_sum+=std::abs(v*t.a);++stats.edge_nodes;
        }
    }
    return t.a*sum.value;
}
py::dict stats_dict(const Stats& s) {
    py::dict d;
    d["triangle_target_pairs"]=s.pairs;d["regularized_pairs"]=s.regularized;
    d["edge_nodes"]=s.edge_nodes;d["maximum_edge_subdivisions"]=s.max_subdivisions;
    d["sum_absolute_edge_terms_before_fresnel_prefactor"]=s.absolute_edge_sum;
    d["common_longitudinal_phase_omitted"]=true;
    d["arithmetic"]="float64 with compensated complex sums";
    d["error_note"]="Quadrature/roundoff diagnostics are not a certified bound; compare orders and phase_step.";
    return d;
}
template<class T>py::bytes raw_bytes(const std::vector<T>& a) {
    return py::bytes(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));
}
py::dict eval_contour(const std::vector<std::vector<double>>& records,double k,double distance,
                     const std::vector<double>& xs,const std::vector<double>& ys,double width,
                     int order,int pixel_order,double phase_step,std::size_t max_subdivisions,int nthreads) {
    if(!finite(k)||k<=0||!finite(distance)||distance<=0)throw std::invalid_argument("k and distance must be positive finite");
    if(!finite(width)||width<0||!finite(phase_step)||phase_step<=0||!max_subdivisions||nthreads<1)
        throw std::invalid_argument("invalid width, phase_step, max_subdivisions or nthreads");
    check_axes(xs,ys);auto ts=parse(records);auto rule=gauss(order),prule=gauss(width?pixel_order:1);
    std::size_t size=xs.size()*ys.size();std::vector<C> field(size);
    int workers=static_cast<int>(std::min<std::size_t>(nthreads,size));
    std::vector<Stats> stats(workers);std::exception_ptr failure;std::mutex mutex;std::atomic<bool> failed{false};
    double kappa=k/distance;
    if(!finite(kappa)||kappa<=0)throw std::invalid_argument("k/distance must be representable and positive");
    C prefactor=kappa/(2*PI*I);
    {
        py::gil_scoped_release release;
        auto work=[&](int tid){try {
            for(std::size_t n=tid;n<size&&!failed;n+=workers) {
                V center{xs[n%xs.size()],ys[n/xs.size()]};Sum target;
                for(std::size_t iy=0;iy<prule.x.size();++iy)for(std::size_t ix=0;ix<prule.x.size();++ix){
                    V p=center+V{(prule.x[ix]-.5)*width,(prule.x[iy]-.5)*width};Sum point;
                    for(const auto& t:ts)point.add(triangle_fresnel(t,p,kappa,rule,phase_step,max_subdivisions,stats[tid]));
                    target.add(point.value*(prule.w[ix]*prule.w[iy]));
                }
                field[n]=prefactor*target.value;
            }
        }catch(...){std::lock_guard<std::mutex>lock(mutex);if(!failure)failure=std::current_exception();failed=true;}};
        std::vector<std::thread> threads;
        for(int i=1;i<workers;++i)threads.emplace_back(work,i);
        work(0);for(auto& thread:threads)thread.join();
    }
    if(failure)std::rethrow_exception(failure);
    Stats total;for(const auto& s:stats)total.add(s);
    py::dict result;result["field"]=raw_bytes(field);result["shape"]=py::make_tuple(ys.size(),xs.size());
    result["diagnostics"]=stats_dict(total);return result;
}
using Polygon=std::vector<V>;
Polygon clip(const Polygon& input,int axis,double bound,bool keep_greater) {
    Polygon out;if(input.empty())return out;
    auto coord=[axis](V q){return axis?q.y:q.x;};
    auto inside=[&](V q){return keep_greater?coord(q)>=bound:coord(q)<=bound;};
    V a=input.back();bool ai=inside(a);
    for(V b:input){bool bi=inside(b);if(ai!=bi){double f=(bound-coord(a))/(coord(b)-coord(a));out.push_back(a+(b-a)*f);}if(bi)out.push_back(b);a=b;ai=bi;}
    return out;
}
double polygon_area(const Polygon& p) {
    if(p.size()<3)return 0;double det=0;
    for(std::size_t j=1;j+1<p.size();++j)det+=cross(p[j]-p[0],p[j+1]-p[0]);
    return det/2;
}
// Analytic Taylor moments for the removable zero-frequency limit.
C small_linear_triangle(V a,V b,V c,const Triangle& t) {
    double det=cross(b-a,c-a);if(det==0)return {};
    C u=I*dot(t.g,b-a),v=I*dot(t.g,c-a),hm2=0,hm1=1;
    double factorial=2;Sum sum;sum.add(C(.5,0));
    for(int n=1;n<=24;++n){C h=(u+v)*hm1-u*v*hm2;factorial*=n+2;sum.add(h/factorial);hm2=hm1;hm1=h;}
    return det*cis(t.phase+dot(t.g,a-t.anchor))*sum.value;
}
C linear_polygon(const Polygon& p,const Triangle& t) {
    if(p.size()<3)return {};double g2=norm2(t.g),extent=0;
    for(V q:p)extent=std::max(extent,std::abs(dot(t.g,q-p[0])));
    if(extent<.5) {
        Sum s;for(std::size_t j=1;j+1<p.size();++j)s.add(small_linear_triangle(p[0],p[j],p[j+1],t));return t.a*s.value;
    }
    Sum s;
    for(std::size_t j=0;j<p.size();++j){V a=p[j],edge=p[(j+1)%p.size()]-a,normal{edge.y,-edge.x};
        double along=dot(t.g,edge),phase=t.phase+dot(t.g,a-t.anchor)+along/2;
        s.add(cis(phase)*sinc(along/2)*dot(t.g,normal)/(I*g2));}
    return t.a*s.value;
}
bool contains_half_open(const Triangle& t,V p) {
    for(int j=0;j<3;++j){V a=t.q[j],e=t.q[(j+1)%3]-a;double z=cross(e,p-a);
        if(z<0)return false;
        if(z==0&&!(e.y>0||(e.y==0&&e.x<0)))return false;
    }return true;
}
py::dict eval_exit(const std::vector<std::vector<double>>& records,const std::vector<double>& xs,
                   const std::vector<double>& ys,double width) {
    if(!finite(width)||width<0)throw std::invalid_argument("pixel_width must be finite and nonnegative");
    check_axes(xs,ys);auto ts=parse(records);std::vector<Sum> sums(xs.size()*ys.size());
    std::uint64_t intersections=0;
    {
        py::gil_scoped_release release;
        for(const auto& t:ts){double xmin=t.q[0].x,xmax=xmin,ymin=t.q[0].y,ymax=ymin;
            for(V q:t.q){xmin=std::min(xmin,q.x);xmax=std::max(xmax,q.x);ymin=std::min(ymin,q.y);ymax=std::max(ymax,q.y);}
            auto x0=std::lower_bound(xs.begin(),xs.end(),xmin-width/2)-xs.begin(),x1=std::upper_bound(xs.begin(),xs.end(),xmax+width/2)-xs.begin();
            auto y0=std::lower_bound(ys.begin(),ys.end(),ymin-width/2)-ys.begin(),y1=std::upper_bound(ys.begin(),ys.end(),ymax+width/2)-ys.begin();
            for(auto iy=y0;iy<y1;++iy)for(auto ix=x0;ix<x1;++ix){V p{xs[ix],ys[iy]};C val;
                if(width){Polygon poly(t.q.begin(),t.q.end());poly=clip(poly,0,p.x-width/2,true);poly=clip(poly,0,p.x+width/2,false);poly=clip(poly,1,p.y-width/2,true);poly=clip(poly,1,p.y+width/2,false);
                    if(polygon_area(poly)<=0)continue;val=linear_polygon(poly,t)/(width*width);
                }else{if(!contains_half_open(t,p))continue;val=t.a*cis(t.phase+dot(t.g,p-t.anchor));}
                sums[iy*xs.size()+ix].add(val);++intersections;
            }
        }
    }
    std::vector<C> field;field.reserve(sums.size());for(const auto& s:sums)field.push_back(s.value);
    py::dict diagnostic;diagnostic["triangle_receiver_intersections"]=intersections;
    diagnostic["pixel_normalization"]="full square area; field outside triangles is zero";
    diagnostic["point_boundary_convention"]="half-open oriented edges; each regular shared edge has one owner";
    diagnostic["linear_phase_integration"]="analytic contour, analytic Taylor moments near zero frequency";
    py::dict result;result["field"]=raw_bytes(field);result["shape"]=py::make_tuple(ys.size(),xs.size());result["diagnostics"]=diagnostic;return result;
}
template<class T>std::vector<T> buffer_copy(const py::buffer& value,std::size_t size) {
    auto info=value.request();
    if(static_cast<std::size_t>(info.size*info.itemsize)!=size*sizeof(T))
        throw std::invalid_argument("buffer has wrong byte length");
    if(info.ndim!=1||info.strides[0]!=info.itemsize)throw std::invalid_argument("buffer must be contiguous and one-dimensional");
    std::vector<T> out(size);std::memcpy(out.data(),info.ptr,size*sizeof(T));return out;
}
class CoherenceAccumulator {
    std::size_t size_,ref_;
    std::uint64_t count_=0;
    std::vector<double> intensity_,intensity_comp_;
    std::vector<C> w_,w_comp_;
    double ref_intensity_=0,ref_comp_=0;
public:
    CoherenceAccumulator(std::size_t size,std::size_t reference):size_(size),ref_(reference),
        intensity_(size),intensity_comp_(size),w_(size),w_comp_(size) {
        if(!size||reference>=size)throw std::invalid_argument("reference index must be inside nonempty field");
    }
    void add(const py::buffer& buffer) {
        auto field=buffer_copy<C>(buffer,size_);
        py::gil_scoped_release release;
        for(C z:field)if(!finite(z.real())||!finite(z.imag()))throw std::invalid_argument("field contains a non-finite value");
        C reference=field[ref_];double ri=std::norm(reference);
        if(!finite(ri))throw std::invalid_argument("field intensity overflow");
        for(C z:field)if(!finite(std::norm(z))||!finite((z*std::conj(reference)).real())||!finite((z*std::conj(reference)).imag()))
            throw std::invalid_argument("field moments overflow");
        for(std::size_t n=0;n<size_;++n){
            double y=std::norm(field[n])-intensity_comp_[n],sum=intensity_[n]+y;
            intensity_comp_[n]=(sum-intensity_[n])-y;intensity_[n]=sum;
            C wy=field[n]*std::conj(reference)-w_comp_[n],ws=w_[n]+wy;
            w_comp_[n]=(ws-w_[n])-wy;w_[n]=ws;
        }
        double y=ri-ref_comp_,sum=ref_intensity_+y;ref_comp_=(sum-ref_intensity_)-y;ref_intensity_=sum;++count_;
    }
    py::dict result()const {
        if(!count_)throw std::runtime_error("cannot normalize an empty ensemble");
        std::vector<double> intensity(size_);std::vector<C> w(size_),mu(size_);std::vector<unsigned char> valid(size_);
        {
            py::gil_scoped_release release;
            for(std::size_t n=0;n<size_;++n){intensity[n]=intensity_[n]/count_;w[n]=w_[n]/static_cast<double>(count_);
                if(intensity_[n]>0&&ref_intensity_>0){mu[n]=w_[n]/std::sqrt(intensity_[n])/std::sqrt(ref_intensity_);valid[n]=1;}
                else mu[n]=C(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN());
            }
        }
        py::dict d;d["I"]=raw_bytes(intensity);d["W"]=raw_bytes(w);d["mu"]=raw_bytes(mu);d["valid"]=raw_bytes(valid);
        d["reference_intensity"]=ref_intensity_/count_;d["n_modes"]=count_;d["ref_index"]=ref_;return d;
    }
    py::dict state()const {
        py::dict d;d["size"]=size_;d["ref_index"]=ref_;d["n_modes"]=count_;
        d["I_sum"]=raw_bytes(intensity_);d["I_comp"]=raw_bytes(intensity_comp_);
        d["W_sum"]=raw_bytes(w_);d["W_comp"]=raw_bytes(w_comp_);
        d["ref_sum"]=ref_intensity_;d["ref_comp"]=ref_comp_;return d;
    }
    static CoherenceAccumulator from_state(const py::dict& d) {
        CoherenceAccumulator a(d["size"].cast<std::size_t>(),d["ref_index"].cast<std::size_t>());
        a.count_=d["n_modes"].cast<std::uint64_t>();
        a.intensity_=buffer_copy<double>(d["I_sum"].cast<py::buffer>(),a.size_);
        a.intensity_comp_=buffer_copy<double>(d["I_comp"].cast<py::buffer>(),a.size_);
        a.w_=buffer_copy<C>(d["W_sum"].cast<py::buffer>(),a.size_);
        a.w_comp_=buffer_copy<C>(d["W_comp"].cast<py::buffer>(),a.size_);
        a.ref_intensity_=d["ref_sum"].cast<double>();a.ref_comp_=d["ref_comp"].cast<double>();return a;
    }
};
}
PYBIND11_MODULE(_stage19,m) {
    m.doc()="NumPy-free constant-amplitude linear-phase triangle integration for Stage 19.";
    m.def("eval_contour",&eval_contour,py::arg("triangles"),py::arg("k"),py::arg("distance"),
          py::arg("x_centers"),py::arg("y_centers"),py::arg("pixel_width")=0.,py::arg("order")=8,
          py::arg("pixel_order")=4,py::arg("phase_step")=4.,py::arg("max_subdivisions")=100000,py::arg("nthreads")=1);
    m.def("eval_exit",&eval_exit,py::arg("triangles"),py::arg("x_centers"),py::arg("y_centers"),py::arg("pixel_width")=0.);
    m.attr("record_layout")="x0,y0,x1,y1,x2,y2,anchor_x,anchor_y,phase_gradient_x,phase_gradient_y,A_re,A_im,phase_rad";
    py::class_<CoherenceAccumulator>(m,"CoherenceAccumulator")
        .def(py::init<std::size_t,std::size_t>(),py::arg("size"),py::arg("ref_index"))
        .def("add",&CoherenceAccumulator::add,py::arg("field"))
        .def("result",&CoherenceAccumulator::result)
        .def("state",&CoherenceAccumulator::state)
        .def_static("from_state",&CoherenceAccumulator::from_state);
    bind_stage19_fast(m);
}
