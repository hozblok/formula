#ifndef EVAL_MPF_H
#define EVAL_MPF_H

#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/variant.hpp>

// TODO: uncomment to debug
// #ifndef CSDEBUG
// #include <iostream>
// #define CSDEBUG
// #endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 8198 digits of pi number.
#ifndef M_PI_STR
#define M_PI_STR                                                               \
  "3."                                                                         \
  "14159265358979323846264338327950288419716939937510582097494459230781640628" \
  "62089986280348253421170679821480865132823066470938446095505822317253594081" \
  "28481117450284102701938521105559644622948954930381964428810975665933446128" \
  "47564823378678316527120190914564856692346034861045432664821339360726024914" \
  "12737245870066063155881748815209209628292540917153643678925903600113305305" \
  "48820466521384146951941511609433057270365759591953092186117381932611793105" \
  "11854807446237996274956735188575272489122793818301194912983367336244065664" \
  "30860213949463952247371907021798609437027705392171762931767523846748184676" \
  "69405132000568127145263560827785771342757789609173637178721468440901224953" \
  "43014654958537105079227968925892354201995611212902196086403441815981362977" \
  "47713099605187072113499999983729780499510597317328160963185950244594553469" \
  "08302642522308253344685035261931188171010003137838752886587533208381420617" \
  "17766914730359825349042875546873115956286388235378759375195778185778053217" \
  "12268066130019278766111959092164201989380952572010654858632788659361533818" \
  "27968230301952035301852968995773622599413891249721775283479131515574857242" \
  "45415069595082953311686172785588907509838175463746493931925506040092770167" \
  "11390098488240128583616035637076601047101819429555961989467678374494482553" \
  "79774726847104047534646208046684259069491293313677028989152104752162056966" \
  "02405803815019351125338243003558764024749647326391419927260426992279678235" \
  "47816360093417216412199245863150302861829745557067498385054945885869269956" \
  "90927210797509302955321165344987202755960236480665499119881834797753566369" \
  "80742654252786255181841757467289097777279380008164706001614524919217321721" \
  "47723501414419735685481613611573525521334757418494684385233239073941433345" \
  "47762416862518983569485562099219222184272550254256887671790494601653466804" \
  "98862723279178608578438382796797668145410095388378636095068006422512520511" \
  "73929848960841284886269456042419652850222106611863067442786220391949450471" \
  "23713786960956364371917287467764657573962413890865832645995813390478027590" \
  "09946576407895126946839835259570982582262052248940772671947826848260147699" \
  "09026401363944374553050682034962524517493996514314298091906592509372216964" \
  "61515709858387410597885959772975498930161753928468138268683868942774155991" \
  "85592524595395943104997252468084598727364469584865383673622262609912460805" \
  "12438843904512441365497627807977156914359977001296160894416948685558484063" \
  "53422072225828488648158456028506016842739452267467678895252138522549954666" \
  "72782398645659611635488623057745649803559363456817432411251507606947945109" \
  "65960940252288797108931456691368672287489405601015033086179286809208747609" \
  "17824938589009714909675985261365549781893129784821682998948722658804857564" \
  "01427047755513237964145152374623436454285844479526586782105114135473573952" \
  "31134271661021359695362314429524849371871101457654035902799344037420073105" \
  "78539062198387447808478489683321445713868751943506430218453191048481005370" \
  "61468067491927819119793995206141966342875444064374512371819217999839101591" \
  "95618146751426912397489409071864942319615679452080951465502252316038819301" \
  "42093762137855956638937787083039069792077346722182562599661501421503068038" \
  "44773454920260541466592520149744285073251866600213243408819071048633173464" \
  "96514539057962685610055081066587969981635747363840525714591028970641401109" \
  "71206280439039759515677157700420337869936007230558763176359421873125147120" \
  "53292819182618612586732157919841484882916447060957527069572209175671167229" \
  "10981690915280173506712748583222871835209353965725121083579151369882091444" \
  "21006751033467110314126711136990865851639831501970165151168517143765761835" \
  "15565088490998985998238734552833163550764791853589322618548963213293308985" \
  "70642046752590709154814165498594616371802709819943099244889575712828905923" \
  "23326097299712084433573265489382391193259746366730583604142813883032038249" \
  "03758985243744170291327656180937734440307074692112019130203303801976211011" \
  "00449293215160842444859637669838952286847831235526582131449576857262433441" \
  "89303968642624341077322697802807318915441101044682325271620105265227211166" \
  "03966655730925471105578537634668206531098965269186205647693125705863566201" \
  "85581007293606598764861179104533488503461136576867532494416680396265797877" \
  "18556084552965412665408530614344431858676975145661406800700237877659134401" \
  "71274947042056223053899456131407112700040785473326993908145466464588079727" \
  "08266830634328587856983052358089330657574067954571637752542021149557615814" \
  "00250126228594130216471550979259230990796547376125517656751357517829666454" \
  "77917450112996148903046399471329621073404375189573596145890193897131117904" \
  "29782856475032031986915140287080859904801094121472213179476477726224142548" \
  "54540332157185306142288137585043063321751829798662237172159160771669254748" \
  "73898665494945011465406284336639379003976926567214638530673609657120918076" \
  "38327166416274888800786925602902284721040317211860820419000422966171196377" \
  "92133757511495950156604963186294726547364252308177036751590673502350728354" \
  "05670403867435136222247715891504953098444893330963408780769325993978054193" \
  "41447377441842631298608099888687413260472156951623965864573021631598193195" \
  "16735381297416772947867242292465436680098067692823828068996400482435403701" \
  "41631496589794092432378969070697794223625082216889573837986230015937764716" \
  "51228935786015881617557829735233446042815126272037343146531977774160319906" \
  "65541876397929334419521541341899485444734567383162499341913181480927777103" \
  "86387734317720754565453220777092120190516609628049092636019759882816133231" \
  "66636528619326686336062735676303544776280350450777235547105859548702790814" \
  "35624014517180624643626794561275318134078330336254232783944975382437205835" \
  "31147711992606381334677687969597030983391307710987040859133746414428227726" \
  "34659470474587847787201927715280731767907707157213444730605700733492436931" \
  "13835049316312840425121925651798069411352801314701304781643788518529092854" \
  "52011658393419656213491434159562586586557055269049652098580338507224264829" \
  "39728584783163057777560688876446248246857926039535277348030480290058760758" \
  "25104747091643961362676044925627420420832085661190625454337213153595845068" \
  "77246029016187667952406163425225771954291629919306455377991403734043287526" \
  "28889639958794757291746426357455254079091451357111369410911939325191076020" \
  "82520261879853188770584297259167781314969900901921169717372784768472686084" \
  "90033770242429165130050051683233643503895170298939223345172201381280696501" \
  "17844087451960121228599371623130171144484640903890644954440061986907548516" \
  "02632750529834918740786680881833851022833450850486082503930213321971551843" \
  "06354550076682829493041377655279397517546139539846833936383047461199665385" \
  "81538420568533862186725233402830871123282789212507712629463229563989898935" \
  "82116745627010218356462201349671518819097303811980049734072396103685406643" \
  "19395097901906996395524530054505806855019567302292191393391856803449039820" \
  "59551002263535361920419947455385938102343955449597783779023742161727111723" \
  "64343543947822181852862408514006660443325888569867054315470696574745855033" \
  "23233421073015459405165537906866273337995851156257843229882737231989875714" \
  "15957811196358330059408730681216028764962867446047746491599505497374256269" \
  "01049037781986835938146574126804925648798556145372347867330390468838343634" \
  "65537949864192705638729317487233208376011230299113679386270894387993620162" \
  "95154133714248928307220126901475466847653576164773794675200490757155527819" \
  "65362132392640616013635815590742202020318727760527721900556148425551879253" \
  "03435139844253223415762336106425063904975008656271095359194658975141310348" \
  "22769306247435363256916078154781811528436679570611086153315044521274739245" \
  "44945423682886061340841486377670096120715124914043027253860764823634143346" \
  "23518975766452164137679690314950191085759844239198629164219399490723623464" \
  "68441173940326591840443780513338945257423995082965912285085558215725031071" \
  "25701266830240292952522011872676756220415420516184163484756516999811614101" \
  "00299607838690929160302884002691041407928862150784245167090870006992821206" \
  "60418371806535567252532567532861291042487761825829765157959847035622262934" \
  "86003415872298053498965022629174878820273420922224533985626476691490556284" \
  "25039127577102840279980663658254889264880254566101729670266407655904290994" \
  "56815065265305371829412703369313785178609040708667114965583434347693385781" \
  "71138645587367812301458768712660348913909562009939361031"
#endif

const size_t kNpos = static_cast<size_t> (~0);

/**
 * To check that substring is the number.
 * e.g. -1e+10, 0009., -.1, etc.
 */
const std::regex kIsNumberRegex(
    R"(^([+-]?(?:[[:d:]]+\.?|[[:d:]]*\.[[:d:]]+))(?:[Ee][+-]?[[:d:]]+)?$)");
const std::string kNumberSymbols("+-0123456789.eE");
/**
 * Operations: logical or, logical and, relational =, <, >, addition,
 * subtraction, multiplication, division, the construction of the power.
 */
const std::string kOperations("|&=><+-*/^");
const std::string kOperationsWithParentheses("()|&=><+-*/^");

/** '_', any letters are allowed. */
// ValueError: Variable name '.e+0' contains forbidden characters: +
// We can allow all except ()|&=><+-*/^, but... Is it necessary?
// cannot start with numbers and '.'. (Formula("pi + 0e + 0").get({'e1': '1'}))
// Formula("pi + .e + 0").get({'.e':'30'})
const std::string kForbiddenVariableSymbols("()[]{}~!@#$%\\?,;`'\"|&=><+*/^-");
const std::string kForbiddenStartVariableSymbols(".0123456789");
const std::regex kWrongVariableRegex(
    R"([\(\)\[\]\{\}~!@#\$%\\\?,;`'\"\|&=><\+\*/\^-])");

/**
 * Arbitrary-precision arithmetic.
 */
template <unsigned N>
using mp_real =
    boost::multiprecision::number<boost::multiprecision::cpp_dec_float<N>,
                                  boost::multiprecision::et_on>;

/**
 * All available precisions for high accuracy calculations.
 */
enum AllowedPrecisions : unsigned {
  p_16 = 16U,
  p_24 = 24U,
  p_32 = 32U,
  p_48 = 48U,
  p_64 = 64U,
  p_96 = 96U,
  p_128 = 128U,
  p_192 = 192U,
  p_256 = 256U,
  p_384 = 384U,
  p_512 = 512U,
  p_768 = 768U,
  p_1024 = 1024U,
  p_2048 = 2048U,
  p_3072 = 3072U,
  p_4096 = 4096U,
  p_6144 = 6144U,
  p_8192 = 8192U,
};

// TODO support more precision options.
/**
 * Allowed precisions sorted in ascending order.
 */
static const AllowedPrecisions min_precision = p_16;
static const AllowedPrecisions max_precision = p_8192;
static const std::tuple<AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions>
    precisions = std::make_tuple(p_16, p_24, p_32, p_48, p_64, p_96, p_128,
                                 p_192, p_256, p_384, p_512, p_768, p_1024,
                                 p_2048, p_3072, p_4096, p_6144, p_8192);
constexpr unsigned kPrecisionsLength = 18U;
constexpr unsigned precisions_array[kPrecisionsLength] = {
    16U,  24U,  32U,  48U,   64U,   96U,   128U,  192U,  256U,
    384U, 512U, 768U, 1024U, 2048U, 3072U, 4096U, 6144U, 8192U};

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type for_each(
    const std::tuple<Tp...> &, FuncT)  // Unused arguments are given no names.
{}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type for_each(const std::tuple<Tp...> &t, FuncT f) {
  bool next = f(std::get<I>(t));
  if (next) {
    for_each<I + 1, FuncT, Tp...>(t, f);
  }
}

// TODO Complex numbers
// typedef std::complex<float100et> complexFloat100et;

using boost::math::constants::pi;
using boost::multiprecision::acos;
using boost::multiprecision::asin;
using boost::multiprecision::atan;
using boost::multiprecision::cos;
using boost::multiprecision::exp;
using boost::multiprecision::fabs;
using boost::multiprecision::log;
using boost::multiprecision::pow;
using boost::multiprecision::sin;
using boost::multiprecision::sqrt;
using boost::multiprecision::swap;
using boost::multiprecision::tan;

/**
 * Class for evaluating formula specified by the string
 * typename Real (not mp_real<NUMBER>) for double support.
 */
template <typename Real>
class cseval {
  /**
   * Kind of formula element:
   * 'n' - number, 'v' - variable, 'f' - function, 'e' - error.
   */
  char kind_;

  /**
   * The function name or variable name for current node. e.g.
   * "+","-","/","x","a"
   * (!) all variable names must be represented by one Latin letter (!).
   * (!) i, j - reserved for complex numbers. (!)
   */
  std::string id_;

  /**
   * Value which was parset from expression, value makes sense if kind is 'n'.
   */
  Real value_;

  /** Child elements of the formula tree. */
  cseval *left_eval_, *right_eval_;

  /** Symbol of the imaginary unit (by default - 'i'). */
  char imaginary_unit_;

  /** Try to find each symbol of {symbols} string in the {str} string. */
  std::unordered_map<char, size_t> operations_outside_parentheses(
      const std::string &str, const std::string &symbols) const;

  /**  Try to find symbols outside parentheses: '(' and ')'. */
  bool isThereSymbolsOutsideParentheses(const std::string &str) const;

 public:
  cseval(std::string expression, char imaginary_unit = 'i');

  cseval(const cseval &other);

  ~cseval();

  cseval &operator=(const cseval &other) {
    if (this != &other) {
      kind_ = other.kind_;
      id_ = std::string(other.id_);
      value_ = Real(other.value_);
      imaginary_unit_ = other.imaginary_unit_;
      if (left_eval_) {
        delete left_eval_;
      }
      if (other.left_eval_) {
        left_eval_ = new cseval<Real>(other.left_eval_);
      } else {
        left_eval_ = nullptr;
      }
      if (right_eval_) {
        delete right_eval_;
      }
      if (other.right_eval_) {
        right_eval_ = new cseval<Real>(other.right_eval_);
      } else {
        right_eval_ = nullptr;
      }
    }
    return *this;
  }

  void collect_variables(std::unordered_set<std::string> *variables) {
    if (left_eval_) {
      left_eval_->collect_variables(variables);
    }
    if (right_eval_) {
      right_eval_->collect_variables(variables);
    }
    if (kind_ == 'v') {
#ifdef CSDEBUG
      std::cout << "collect_variables, kind:" << kind_ << ", id:" << id_
                << std::endl;
#endif
      variables->insert(std::string(id_));
    }
  }

  // Evaluation of subformula.
  Real calculate(const std::map<std::string, Real> &variables_to_values,
                 const std::map<std::string, Real (*)(Real, Real)>
                     &mapFunctionTwoArgsValue = functionsTwoArgs,
                 const std::map<std::string, Real (*)(Real)>
                     &mapFunctionOneArgValue = functionsOneArg) const;
  Real calculate(const std::map<std::string, std::string> &variables_to_values,
                 const std::map<std::string, Real (*)(Real, Real)>
                     &mapFunctionTwoArgsValue = functionsTwoArgs,
                 const std::map<std::string, Real (*)(Real)>
                     &mapFunctionOneArgValue = functionsOneArg) const;
  Real calculate(const std::map<std::string, double> &variables_to_values,
                 const std::map<std::string, Real (*)(Real, Real)>
                     &mapFunctionTwoArgsValue = functionsTwoArgs,
                 const std::map<std::string, Real (*)(Real)>
                     &mapFunctionOneArgValue = functionsOneArg) const;
  // Evaluation derivative of subformula.
  Real calculate_derivative(
      const std::string &variable,
      const std::map<std::string, Real> &variables_to_values,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          functionsOneArg,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft =
          functionsTwoArgsDLeft,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight =
          functionsTwoArgsDRight) const;

  Real calculate_derivative(
      const std::string &variable,
      const std::map<std::string, std::string> &variables_to_values,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          functionsOneArg,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft =
          functionsTwoArgsDLeft,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight =
          functionsTwoArgsDRight) const;

  // Usefull constants.
  const static Real ZERO;
  const static Real ONE;

  //+ general static methods that have a one-to-one correspondence
  // with the operations specified in the expression string
  // "+" - addition
  static Real _add(Real a, Real b) { return a + b; }
  // "-" - subtraction
  static Real _sub(Real a, Real b) { return a - b; }
  // "&" - and
  static Real _and(Real a, Real b) {
    return a != ZERO && b != ZERO ? ONE : ZERO;
  }
  // "|" - or
  static Real _or(Real a, Real b) {
    return a != ZERO || b != ZERO ? ONE : ZERO;
  }
  // "=" - is equal to
  static Real _eq(Real a, Real b) { return a == b ? ONE : ZERO; }
  // "<"
  static Real _lt(Real a, Real b) { return a < b ? ONE : ZERO; }
  // ">"
  static Real _gt(Real a, Real b) { return a > b ? ONE : ZERO; }
  // "/" - division
  static Real _truediv(Real a, Real b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the \'/\' operation");
    }
    return a / b;
  }
  // division for the computation of the derivative (left path)
  static Real _truediv1(Real, Real b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation the left path \
of the derivative");
    }
    return 1 / b;
  }
  // division for the computation of the derivative (right path)
  static Real _truediv2(Real a, Real b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
right path of the derivative");
    }
    return ZERO - a / (b * b);
  }
  // "*" - multiplication
  static Real _mul(Real a, Real b) { return a * b; }
  // multiplication for the computation of the derivative (left path)
  static Real _mul1(Real, Real b) { return b; }
  // multiplication for the computation of the derivative (right path)
  static Real _mul2(Real a, Real) { return a; }
  // "^" - exponentiation
  static Real _pow(Real a, Real b) { return pow(a, b); }
  // exponentiation for the computation of the derivative (left path)
  static Real _pow1(Real a, Real b) { return (b * _pow(a, b - ONE)); }
  // exponentiation for the computation of the derivative (right path)
  // TODO test log()
  static Real _pow2(Real a, Real b) { return (_log(a) * _pow(a, b)); }
  //- general static methods

  //+ trigonometric functions, exp, log, sqrt and methods for the computation of
  // the derivative:
  // "sin"
  static Real _sin(Real a) { return sin(a); }
  // "sin" for the derivative
  static Real _sin_d(Real a, Real) { return cos(a); }
  // "asin"
  static Real _asin(Real a) { return asin(a); }
  // "asin" for the derivative
  static Real _asin_d(Real a, Real) {
    if (a * a == ONE) {
      // TODO may be inf?
      throw std::invalid_argument(
          "Division by zero during the computation of the arcsin derivative");
    }
    return ONE / _sqrt(ONE - a * a);
  }
  // "cos"
  static Real _cos(Real a) { return cos(a); }
  // "cos" for the derivative
  static Real _cos_d(Real a, Real) { return ZERO - _sin(a); }
  // "acos"
  static Real _acos(Real a) { return acos(a); }
  static Real _acos_d(Real a, Real) {
    if (a * a == ONE) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the arccos derivative");
    }
    return ZERO - (ONE / _sqrt(ONE - a * a));
  }
  // "tan"
  static Real _tan(Real a) { return tan(a); }
  // "tan" for the derivative
  static Real _tan_d(Real a, Real) {
    if (_cos(a) == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the tangent derivative");
    }
    return ONE / (_cos(a) * _cos(a));
  }
  // "atan"
  static Real _atan(Real a) { return atan(a); }
  // "atan" for the derivative
  static Real _atan_d(Real a, Real) { return ONE / (ONE + a * a); }
  // "exp"
  static Real _exp(Real a) { return exp(a); }
  // "exp" for the derivative
  static Real _exp_d(Real a, Real) { return exp(a); }
  // "log" - (!) Natural logarithm
  static Real _log(Real a) { return log(a); }
  // "log" - (!) Natural logarithm for the derivative
  static Real _log_d(Real a, Real) {
    if (a == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the natural logarithm derivative");
    }
    return ONE / a;
  }
  // "sqrt" - square root
  static Real _sqrt(Real a) { return sqrt(a); }
  // TODO test _sqrt_d()
  // "sqrt" for the derivative
  static Real _sqrt_d(Real a, Real) {
    if (sqrt(a) == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the sqrt derivative");
    }
    return ONE / (2 * sqrt(a));
  }
  //- trigonometric functions, exp, log, sqrt

  //+ auxiliary methods for the computation of the derivative:
  static Real _zero(Real, Real) { return ZERO; }
  static Real _one(Real, Real) { return ONE; }
  static Real _m_one(Real, Real) { return ZERO - ONE; }
  //- auxiliary methods

  // dictionaries contain the appropriate names of operations and static methods
  // for evaluating, e.g.
  // "+" -> _add
  // "sin" -> _sin
  static const std::map<std::string, Real (*)(Real, Real)> functionsTwoArgs;
  static const std::map<std::string, Real (*)(Real)> functionsOneArg;

  // dictionaries contain references to derivatives of basic functions and their
  // names:
  static const std::map<std::string, Real (*)(Real, Real)>
      functionsTwoArgsDLeft;
  static const std::map<std::string, Real (*)(Real, Real)>
      functionsTwoArgsDRight;
  static const std::map<std::string, Real (*)(Real)> functionsOneArgD;
};

#endif  // EVAL_MPF_H
