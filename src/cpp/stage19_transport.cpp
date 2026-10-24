// Research/prototype interval certificate for Stage 19 ray variations.
// Assumptions: IEEE binary64, round-to-nearest, no fast-math/reassociation,
// correctly rounded elementary arithmetic and sqrt. Inputs enclose the real
// MP trace values. This does not certify the GO model or MP trace residuals.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#define STAGE19_HAS_MXCSR 1
#endif

#ifdef __FAST_MATH__
#error Stage 19 interval transport must not be built with fast-math
#endif
#ifdef _MSC_VER
#pragma float_control(precise, on)
#endif

namespace py = pybind11;
namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
struct Uncertain : std::runtime_error { using std::runtime_error::runtime_error; };
void check_environment() {
    if(!std::numeric_limits<double>::is_iec559||std::numeric_limits<double>::digits!=53)
        throw Uncertain("unsupported_binary64");
    if(std::fegetround()!=FE_TONEAREST)throw Uncertain("unsupported_rounding_mode");
#ifdef STAGE19_HAS_MXCSR
    // FTZ (bit 15) and DAZ (bit 6) are independent of the rounding-mode bits.
    if((_mm_getcsr() & 0x8040u)!=0)throw Uncertain("unsupported_ftz_daz");
#endif
    // Portable runtime probe, with volatile to prohibit constant folding.
    volatile double smallest=std::numeric_limits<double>::min();
    volatile double half=smallest*0.5;
    volatile double restored=half*2.0;
    if(half==0.0||restored!=smallest)throw Uncertain("unsupported_gradual_underflow");
}
struct I {
    double lo, hi;
    I(double x=0):lo(x),hi(x) {}
    I(double a,double b):lo(a),hi(b) {
        if (!std::isfinite(a)||!std::isfinite(b)||a>b) throw Uncertain("invalid_interval");
    }
    bool zero() const { return lo==0&&hi==0; }
    bool contains_zero() const { return lo<=0&&hi>=0; }
};
double down(double x) {
    if (!std::isfinite(x)) throw Uncertain("nonfinite_arithmetic");
    double y=std::nextafter(x,-INF);
    if (!std::isfinite(y)) throw Uncertain("interval_overflow");
    return y;
}
double up(double x) {
    if (!std::isfinite(x)) throw Uncertain("nonfinite_arithmetic");
    double y=std::nextafter(x,INF);
    if (!std::isfinite(y)) throw Uncertain("interval_overflow");
    return y;
}
I operator-(I a) { return I(-a.hi,-a.lo); }
I operator+(I a,I b) {
    if(a.zero())return b;if(b.zero())return a;
    return I(down(a.lo+b.lo),up(a.hi+b.hi));
}
I operator-(I a,I b) { return a+(-b); }
I operator*(I a,I b) {
    if(a.zero()||b.zero())return I(0);
    std::array<double,4> x={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
    for(double v:x)if(!std::isfinite(v))throw Uncertain("nonfinite_arithmetic");
    return I(down(*std::min_element(x.begin(),x.end())),up(*std::max_element(x.begin(),x.end())));
}
I operator/(I a,I b) {
    if(b.contains_zero())throw Uncertain("denominator_contains_zero");
    if(a.zero())return I(0);
    return a*I(down(1/b.hi),up(1/b.lo));
}
I sq(I a) {
    if(a.zero())return I(0);
    double low=a.contains_zero()?0:std::min(a.lo*a.lo,a.hi*a.hi);
    double high=std::max(a.lo*a.lo,a.hi*a.hi);
    return I(low==0?0:std::max(0.0,down(low)),up(high));
}
I root(I a) {
    if(a.lo<0)throw Uncertain("sqrt_interval_negative");
    if(a.zero())return I(0);
    return I(a.lo==0?0:std::max(0.0,down(std::sqrt(a.lo))),up(std::sqrt(a.hi)));
}
double minabs(I a) {return a.contains_zero()?0:std::min(std::abs(a.lo),std::abs(a.hi));}
double middle(I a) {return a.lo/2+a.hi/2;}
bool overlap(I a,I b) {return std::max(a.lo,b.lo)<=std::min(a.hi,b.hi);}
using V3=std::array<I,3>;
using M22=std::array<std::array<I,2>,2>;
using M32=std::array<std::array<I,2>,3>;
using M33=std::array<std::array<I,3>,3>;
I dot(V3 a,V3 b) {I x;for(int i=0;i<3;++i)x=x+a[i]*b[i];return x;}
I norm(V3 a) {I x;for(auto v:a)x=x+sq(v);return root(x);}
I det(M22 a) {return a[0][0]*a[1][1]-a[0][1]*a[1][0];}
I read_i(py::handle value) {
    auto s=py::cast<py::sequence>(value);
    if(s.size()!=2)throw Uncertain("interval_requires_two_bounds");
    return I(py::cast<double>(s[0]),py::cast<double>(s[1]));
}
V3 read_v3(py::handle value) {
    auto s=py::cast<py::sequence>(value);
    if(s.size()!=3)throw Uncertain("vector_requires_three_intervals");
    V3 result;for(int i=0;i<3;++i)result[i]=read_i(s[i]);return result;
}
M22 read_m22(py::handle value) {
    auto s=py::cast<py::sequence>(value);M22 result;
    if(s.size()!=2)throw Uncertain("matrix_requires_two_rows");
    for(int i=0;i<2;++i){auto row=py::cast<py::sequence>(s[i]);
        if(row.size()!=2)throw Uncertain("matrix_requires_two_columns");
        for(int j=0;j<2;++j)result[i][j]=read_i(row[j]);}
    return result;
}
py::list export_i(I v) {py::list a;a.append(v.lo);a.append(v.hi);return a;}
py::list export_m(M22 q) {py::list a;for(auto row:q){py::list r;for(I x:row)r.append(export_i(x));a.append(r);}return a;}
struct Segment {
    int count=0;
    std::vector<I> roots;
    std::string method;
};
I polynomial(I a,I b,I c,I t){return (a*t+b)*t+c;}
Segment classify(M22 q,M22 velocity,I distance,I endpoint) {
    M22 v=velocity;for(auto &row:v)for(auto &x:row)x=x*distance;
    I a=det(v),c=det(q);
    I b=q[0][0]*v[1][1]+v[0][0]*q[1][1]-q[0][1]*v[1][0]-v[0][1]*q[1][0];
    // Bernstein enclosure on an EXTENDED interval, so a focus just outside
    // the segment cannot evade the MP endpoint ambiguity convention.
    I t0=-endpoint,t1=I(1)+endpoint;
    I b0=polynomial(a,b,c,t0),b2=polynomial(a,b,c,t1);
    I b1=b0+(t1-t0)*(I(2)*a*t0+b)/I(2);
    if((b0.lo>0&&b1.lo>0&&b2.lo>0)||(b0.hi<0&&b1.hi<0&&b2.hi<0))
        return Segment{0,{},"bernstein_no_root"};
    std::vector<I> roots;
    std::string method;
    if(a.contains_zero()) {
        if(!a.zero())throw Uncertain("quadratic_degree_uncertain");
        if(b.contains_zero())throw Uncertain("linear_degree_uncertain");
        roots.push_back(-c/b);method="linear_interval";
    } else {
        I discriminant=sq(b)-I(4)*a*c;
        if(discriminant.lo<=0)throw Uncertain(discriminant.hi<0?"negative_discriminant":"multiple_focus_uncertain");
        I radical=root(discriminant);
        if(b.lo>0||b.hi<0) {
            I stable=-(b+(b.lo>0?radical:-radical))/I(2);
            if(stable.contains_zero())throw Uncertain("root_division_uncertain");
            roots.push_back(stable/a);roots.push_back(c/stable);
        } else {
            roots.push_back((-b-radical)/(I(2)*a));
            roots.push_back((-b+radical)/(I(2)*a));
        }
        method="quadratic_interval";
    }
    int count=0;
    I interior_upper=I(1)-endpoint,outside_upper=I(1)+endpoint;
    for(auto r:roots) {
        if(r.lo>endpoint.hi&&r.hi<interior_upper.lo)++count;
        else if(r.hi<-endpoint.hi||r.lo>outside_upper.hi){}
        else throw Uncertain("endpoint_focus_uncertain");
    }
    return Segment{count,roots,method};
}
struct Wall {
    V3 normal;
    M33 derivative;
    I radius;
};
struct Bore {
    V3 center;
    I radius;
    bool bent=false;
    I bend_radius;
    V3 tangent,binormal;
};
Bore read_bore(py::dict values) {
    Bore bore;bore.center=read_v3(values["center"]);bore.radius=read_i(values["radius"]);
    if(bore.radius.lo<=0)throw Uncertain("invalid_bore_radius");
    if(values.contains("bend_radius")) {
        bore.bent=true;bore.bend_radius=read_i(values["bend_radius"]);
        if(bore.bend_radius.lo<=0)throw Uncertain("invalid_bend_radius");
        V3 toward=read_v3(values["toward"]);I n=norm(toward);
        if(n.lo<=0)throw Uncertain("undefined_bend_direction");
        for(int i=0;i<3;++i)bore.tangent[i]=toward[i]/n;
        bore.binormal={-bore.tangent[1],bore.tangent[0],I(0)};
    }
    return bore;
}
Wall wall(V3 point,const Bore &bore,I z0) {
    V3 raw;M33 hessian{};
    if(!bore.bent) {
        raw={point[0]-bore.center[0],point[1]-bore.center[1],I(0)};
        hessian[0][0]=I(1);hessian[1][1]=I(1);
    } else {
        V3 local={point[0]-bore.center[0],point[1]-bore.center[1],point[2]-z0};
        I xi=dot(local,bore.tangent),eta=dot(local,bore.binormal),zeta=local[2];
        I R=bore.bend_radius;
        I rho=root(sq(R-xi)+sq(zeta));
        // Rationalized rho-R: never subtract two ~2156 m numbers to obtain
        // a ~24 um cross-sectional radius.
        I dr=(-I(2)*R*xi+sq(xi)+sq(zeta))/(rho+R);
        V3 radial={(xi-R)*bore.tangent[0]/rho,(xi-R)*bore.tangent[1]/rho,zeta/rho};
        for(int i=0;i<3;++i)raw[i]=dr*radial[i]+eta*bore.binormal[i];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
            I rr=radial[i]*radial[j];
            I plane=bore.tangent[i]*bore.tangent[j]+I(i==2&&j==2?1:0);
            hessian[i][j]=rr+(dr/rho)*(plane-rr)+bore.binormal[i]*bore.binormal[j];
        }
    }
    I length=norm(raw);if(length.lo<=0)throw Uncertain("undefined_wall_normal");
    V3 normal;for(int i=0;i<3;++i)normal[i]=raw[i]/length;
    M33 derivative{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int l=0;l<3;++l)
        derivative[i][j]=derivative[i][j]+(I(i==l?1:0)-normal[i]*normal[l])*hessian[l][j]/length;
    return Wall{normal,derivative,length};
}
void reflect(V3 &u,M32 &r,M32 &du,const Wall &wall,I endpoint) {
    I udot=dot(u,wall.normal);
    if(minabs(udot)<=endpoint.hi)throw Uncertain("grazing_uncertain");
    std::array<I,2> dt{},ddot{};M32 hit{},dn{},nextdu{};V3 outgoing;
    for(int j=0;j<2;++j) {
        for(int i=0;i<3;++i)dt[j]=dt[j]-wall.normal[i]*r[i][j];
        dt[j]=dt[j]/udot;
        for(int i=0;i<3;++i)hit[i][j]=r[i][j]+u[i]*dt[j];
        for(int i=0;i<3;++i)for(int l=0;l<3;++l)dn[i][j]=dn[i][j]+wall.derivative[i][l]*hit[l][j];
        for(int i=0;i<3;++i)ddot[j]=ddot[j]+wall.normal[i]*du[i][j]+u[i]*dn[i][j];
    }
    for(int i=0;i<3;++i)outgoing[i]=u[i]-I(2)*udot*wall.normal[i];
    if(outgoing[2].lo<=0)throw Uncertain("forward_direction_uncertain");
    for(int i=0;i<3;++i)for(int j=0;j<2;++j)
        nextdu[i][j]=du[i][j]-I(2)*(wall.normal[i]*ddot[j]+udot*dn[i][j]);
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)
        r[i][j]=hit[i][j]-outgoing[i]*hit[2][j]/outgoing[2];
    r[2]={I(0),I(0)}; // exact fixed-z identity; avoid interval dependency blowup
    u=outgoing;du=nextdu;
}
py::dict certify(py::dict input) {
    py::dict result;result["certified"]=false;result["mp_geometry_verified"]=false;
    result["interval_assumptions"]="IEEE binary64 nearest, outward nextafter, enclosing inputs; GO/trace residuals excluded";
    int maslov=0,segments=0,bernstein=0;
    try {
        check_environment();
        V3 origin=read_v3(input["source"]),start=read_v3(input["entrance"]);
        I z0=read_i(input["z0"]),endpoint=read_i(input["endpoint_tolerance"]),floor=read_i(input["jacobian_floor"]);
        if(endpoint.lo<0||endpoint.hi>=0.1||floor.lo<0)throw Uncertain("invalid_tolerances");
        Bore bore=read_bore(py::cast<py::dict>(input["bore"]));
        auto ps=py::cast<py::sequence>(input["points"]);
        if(ps.size()<1)throw Uncertain("missing_exit_point");
        std::vector<V3> points;for(auto p:ps)points.push_back(read_v3(p));
        V3 final_direction=read_v3(input["direction"]),initial,u;
        for(int i=0;i<3;++i)initial[i]=start[i]-origin[i];
        I distance=norm(initial);if(distance.lo<=0)throw Uncertain("source_on_entrance");
        for(int i=0;i<3;++i)u[i]=initial[i]/distance;
        M32 r{},du{};r[0][0]=I(1);r[1][1]=I(1);
        for(int i=0;i<3;++i)for(int j=0;j<2;++j)du[i][j]=(r[i][j]-u[i]*u[j])/distance;
        py::list focal_intervals;
        for(std::size_t index=0;index<points.size();++index) {
            V3 destination=points[index];I dz=destination[2]-start[2];
            if(dz.lo<0||u[2].lo<=0)throw Uncertain("forward_path_uncertain");
            M22 velocity{},q{{r[0],r[1]}};
            for(int i=0;i<2;++i)for(int j=0;j<2;++j)
                velocity[i][j]=(du[i][j]-u[i]*du[2][j]/u[2])/u[2];
            auto classification=classify(q,velocity,dz,endpoint);
            maslov+=classification.count;++segments;
            if(classification.method=="bernstein_no_root")++bernstein;
            for(I t:classification.roots)if(t.hi>=0&&t.lo<=1) {
                py::dict f;f["segment"]=index;f["t"]=export_i(t);f["z"]=export_i(start[2]+dz*t);focal_intervals.append(f);
            }
            I flight=dz/u[2];
            for(int i=0;i<3;++i) {
                if(!overlap(start[i]+flight*u[i],destination[i]))throw Uncertain("path_interval_nonoverlap");
                for(int j=0;j<2;++j)r[i][j]=r[i][j]+flight*du[i][j];
            }
            if(index+1<points.size()) {
                auto w=wall(destination,bore,z0);
                if(!overlap(w.radius,bore.radius))throw Uncertain("wall_interval_nonoverlap");
                reflect(u,r,du,w,endpoint);
            } else {
                for(int i=0;i<2;++i)for(int j=0;j<2;++j)r[i][j]=r[i][j]-u[i]*r[2][j]/u[2];
                r[2]={I(0),I(0)};
            }
            start=destination;
        }
        for(int i=0;i<3;++i)if(!overlap(u[i],final_direction[i]))throw Uncertain("direction_interval_nonoverlap");
        M22 q{{r[0],r[1]}},p{{du[0],du[1]}};I determinant=det(q);
        if(minabs(determinant)<=floor.hi)throw Uncertain("exit_determinant_uncertain");
        // Exact differential transport is symplectic. Containment of zero is
        // a consistency diagnostic, not the old Decimal residual test.
        I sym=q[0][0]*p[0][1]+q[1][0]*p[1][1]-q[0][1]*p[0][0]-q[1][1]*p[1][0];
        if(!sym.contains_zero())throw Uncertain("lagrangian_interval_nonzero");
        result["certified"]=true;result["reason"]="certified_discrete_transport";
        result["maslov"]=maslov;result["det_sign"]=determinant.lo>0?1:-1;
        result["determinant"]=middle(determinant);result["determinant_interval"]=export_i(determinant);
        result["Q_intervals"]=export_m(q);result["P_intervals"]=export_m(p);
        result["lagrangian_interval"]=export_i(sym);result["focal_intervals"]=focal_intervals;
        result["path_interval_consistent"]=true;
    } catch(const Uncertain &e) {result["reason"]=e.what();}
      catch(const py::cast_error &) {result["reason"]="invalid_payload_type";}
      catch(const py::error_already_set &) {throw;}
    result["segments_certified"]=segments;result["bernstein_segments"]=bernstein;
    return result;
}
py::dict segment_api(py::object q,py::object v,py::object distance,py::object endpoint) {
    py::dict result;result["certified"]=false;
    try {
        check_environment();
        I guard=read_i(endpoint);if(guard.lo<0||guard.hi>=.1)throw Uncertain("invalid_tolerance");
        auto c=classify(read_m22(q),read_m22(v),read_i(distance),guard);
        result["certified"]=true;result["maslov"]=c.count;result["reason"]=c.method;
        py::list roots;for(I r:c.roots)roots.append(export_i(r));result["root_intervals"]=roots;
    } catch(const Uncertain &e) {result["reason"]=e.what();}
    return result;
}
} // namespace

PYBIND11_MODULE(_stage19_transport,m) {
    m.doc()="Prototype outward-interval DRT certificate; MP geometry verification remains separate.";
    m.def("certify",&certify,py::arg("payload"));
    m.def("classify_segment",&segment_api,py::arg("q"),py::arg("velocity_derivative"),
          py::arg("distance"),py::arg("endpoint_tolerance"));
    m.def("environment_status",[](){py::dict result;
        try{check_environment();result["supported"]=true;result["reason"]="ieee_binary64_gradual_nearest";}
        catch(const Uncertain &e){result["supported"]=false;result["reason"]=e.what();}
        return result;});
}
