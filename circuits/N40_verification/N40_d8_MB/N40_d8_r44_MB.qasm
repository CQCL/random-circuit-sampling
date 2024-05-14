OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.780394521715979*pi,1.5814198761304281*pi) q[0];
U1q(1.34451725871686*pi,1.0069652628262162*pi) q[1];
U1q(0.375039060775215*pi,1.81378527385657*pi) q[2];
U1q(3.075983885105893*pi,1.62999494124237*pi) q[3];
U1q(1.3075285720737*pi,0.6201412942947173*pi) q[4];
U1q(0.210271943134193*pi,1.024509218909265*pi) q[5];
U1q(0.748988954002769*pi,1.31749147041813*pi) q[6];
U1q(0.322138350870572*pi,1.243584584607035*pi) q[7];
U1q(0.509376319040715*pi,0.479925184015711*pi) q[8];
U1q(0.584319872359601*pi,1.966115813823653*pi) q[9];
U1q(1.21424214952919*pi,1.6568799869809887*pi) q[10];
U1q(0.34035762450952*pi,0.101225105916254*pi) q[11];
U1q(1.56973730190045*pi,1.7251633932435402*pi) q[12];
U1q(1.63111376063535*pi,1.4077005603453179*pi) q[13];
U1q(1.77895611806152*pi,1.1036151519854003*pi) q[14];
U1q(1.37470399956954*pi,1.798438146270216*pi) q[15];
U1q(1.63884347368798*pi,1.5081264467387183*pi) q[16];
U1q(1.65556627185292*pi,1.5894037731097428*pi) q[17];
U1q(1.73907239504549*pi,0.22457114549939566*pi) q[18];
U1q(0.760296385215456*pi,0.138229585725854*pi) q[19];
U1q(0.333135959309718*pi,0.0349138619260286*pi) q[20];
U1q(0.121523228222043*pi,0.86719484550359*pi) q[21];
U1q(0.617682345071073*pi,1.1223701393203*pi) q[22];
U1q(0.0612598207646955*pi,0.804710228920346*pi) q[23];
U1q(1.38050969254389*pi,0.8840044982502065*pi) q[24];
U1q(3.622188343607556*pi,0.6479046805701293*pi) q[25];
U1q(0.350690416354307*pi,1.6898078872305309*pi) q[26];
U1q(1.67853487249723*pi,0.5488133900908998*pi) q[27];
U1q(1.52081424828409*pi,0.19239356411677336*pi) q[28];
U1q(3.272757490561759*pi,1.1625683104317348*pi) q[29];
U1q(1.35813721230113*pi,1.4606409855751434*pi) q[30];
U1q(0.473203823300821*pi,0.943789824549231*pi) q[31];
U1q(3.249170980292353*pi,1.1612848113640206*pi) q[32];
U1q(1.47626070134231*pi,0.9218401256826995*pi) q[33];
U1q(0.179596702820872*pi,0.64128961184789*pi) q[34];
U1q(0.642601575435758*pi,0.391394954799172*pi) q[35];
U1q(1.86851605662078*pi,0.20585517444501453*pi) q[36];
U1q(0.76189229260038*pi,1.516748960461375*pi) q[37];
U1q(0.699112727590205*pi,1.07504936150057*pi) q[38];
U1q(0.163213063220668*pi,1.746058741938864*pi) q[39];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[30],q[1];
RZZ(0.5*pi) q[2],q[9];
RZZ(0.5*pi) q[3],q[32];
RZZ(0.5*pi) q[15],q[4];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[8],q[34];
RZZ(0.5*pi) q[24],q[10];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[13],q[29];
RZZ(0.5*pi) q[14],q[23];
RZZ(0.5*pi) q[16],q[35];
RZZ(0.5*pi) q[17],q[36];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[19],q[39];
RZZ(0.5*pi) q[21],q[22];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[28],q[37];
U1q(0.537108208206597*pi,0.91450184884448*pi) q[0];
U1q(0.346796456586729*pi,1.4575358571887262*pi) q[1];
U1q(0.640589414046799*pi,0.6340165815951502*pi) q[2];
U1q(0.719446212188746*pi,1.33786476771523*pi) q[3];
U1q(0.745624094706296*pi,1.5478748926994974*pi) q[4];
U1q(0.723432917096755*pi,0.9220507971331999*pi) q[5];
U1q(0.756042696334509*pi,0.41536279461507997*pi) q[6];
U1q(0.497237403910951*pi,1.0455082651536598*pi) q[7];
U1q(0.73613675555967*pi,1.9166581511491252*pi) q[8];
U1q(0.457841419429093*pi,1.30437118256939*pi) q[9];
U1q(0.765407973253706*pi,0.43854854691308875*pi) q[10];
U1q(0.164526113962273*pi,1.0267111323536202*pi) q[11];
U1q(0.395719391307192*pi,1.8011595193345102*pi) q[12];
U1q(0.40665873799656*pi,0.5752270119093579*pi) q[13];
U1q(0.610534578584733*pi,0.47554330163448943*pi) q[14];
U1q(0.736283163760351*pi,1.4199247767934162*pi) q[15];
U1q(0.112763433983643*pi,1.1014233588000781*pi) q[16];
U1q(0.787840606431829*pi,1.2591645777783718*pi) q[17];
U1q(0.656164000177277*pi,0.8230053390486258*pi) q[18];
U1q(0.544303183275877*pi,1.511105918764209*pi) q[19];
U1q(0.249431890087377*pi,1.5078090793396801*pi) q[20];
U1q(0.0893405976383255*pi,1.6495464257076402*pi) q[21];
U1q(0.533943692657556*pi,1.237582332548976*pi) q[22];
U1q(0.251864171651692*pi,0.13120652233007002*pi) q[23];
U1q(0.41036894761538*pi,1.9522091631492815*pi) q[24];
U1q(0.156076511784725*pi,0.6694203429179693*pi) q[25];
U1q(0.468797884526534*pi,1.0071575117186704*pi) q[26];
U1q(0.074645299710487*pi,0.4946670297404596*pi) q[27];
U1q(0.447095177093253*pi,1.1010697845288333*pi) q[28];
U1q(0.408283844210586*pi,0.27427301849817187*pi) q[29];
U1q(0.373357196099449*pi,0.5803953202425336*pi) q[30];
U1q(0.3508648134657*pi,1.1881970574846399*pi) q[31];
U1q(0.88652508582416*pi,1.7479204718141808*pi) q[32];
U1q(0.446508039516097*pi,1.6834327230998993*pi) q[33];
U1q(0.425557880074306*pi,0.33376330301756996*pi) q[34];
U1q(0.250176438631044*pi,0.4345858626188499*pi) q[35];
U1q(0.803843707204404*pi,1.9107235587452447*pi) q[36];
U1q(0.381771064396081*pi,1.0492393700388698*pi) q[37];
U1q(0.823617124614894*pi,1.1168947727221*pi) q[38];
U1q(0.490021828855255*pi,1.97200954117111*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[32],q[4];
RZZ(0.5*pi) q[5],q[20];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[16],q[39];
RZZ(0.5*pi) q[21],q[26];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[28],q[24];
RZZ(0.5*pi) q[38],q[30];
RZZ(0.5*pi) q[34],q[33];
RZZ(0.5*pi) q[37],q[35];
U1q(0.32926919663023*pi,0.5586496671096297*pi) q[0];
U1q(0.386845222149657*pi,1.7252712887857964*pi) q[1];
U1q(0.404886039227992*pi,0.37459408430081975*pi) q[2];
U1q(0.875791223192668*pi,1.048259306688726*pi) q[3];
U1q(0.351665924631206*pi,1.8806791656035369*pi) q[4];
U1q(0.499798268049423*pi,1.83501897730868*pi) q[5];
U1q(0.16630416900981*pi,1.6311831529679797*pi) q[6];
U1q(0.107469917519153*pi,0.47817345138648015*pi) q[7];
U1q(0.579558060382497*pi,1.81311507088493*pi) q[8];
U1q(0.40093385965475*pi,0.8926732149353702*pi) q[9];
U1q(0.85092364872244*pi,0.3075892791561987*pi) q[10];
U1q(0.823582252813157*pi,1.9917165906491396*pi) q[11];
U1q(0.428919964858248*pi,1.9808120107320608*pi) q[12];
U1q(0.317964118930386*pi,1.1369073978271578*pi) q[13];
U1q(0.470614621612143*pi,0.8335202743913803*pi) q[14];
U1q(0.816219012939635*pi,1.905330090210076*pi) q[15];
U1q(0.604153986528317*pi,1.8358461579579481*pi) q[16];
U1q(0.675048467460801*pi,0.6007921840545327*pi) q[17];
U1q(0.203343901760457*pi,1.3452913878802857*pi) q[18];
U1q(0.840990604587761*pi,0.46713842614245005*pi) q[19];
U1q(0.143189974225139*pi,0.8260136085384904*pi) q[20];
U1q(0.108770212331605*pi,1.7323835243324304*pi) q[21];
U1q(0.691325618744228*pi,1.72650388871917*pi) q[22];
U1q(0.2529703725043*pi,1.6859243176039804*pi) q[23];
U1q(0.494893156002058*pi,0.9021303132806966*pi) q[24];
U1q(0.168908845064544*pi,0.7042718000494199*pi) q[25];
U1q(0.592038134636756*pi,1.8716651235931403*pi) q[26];
U1q(0.230269500484292*pi,0.59118294234273*pi) q[27];
U1q(0.290202419930001*pi,0.27830467171120343*pi) q[28];
U1q(0.768141084631242*pi,1.9882004468585954*pi) q[29];
U1q(0.693619348426146*pi,0.2892152603693736*pi) q[30];
U1q(0.308792784761276*pi,0.4111204147258398*pi) q[31];
U1q(0.122173080862754*pi,0.13706830383247048*pi) q[32];
U1q(0.343853152125159*pi,1.5271260756539098*pi) q[33];
U1q(0.539149198430987*pi,1.0823095442867396*pi) q[34];
U1q(0.621811200448577*pi,1.45755985119998*pi) q[35];
U1q(0.708998382473311*pi,1.3590211176796547*pi) q[36];
U1q(0.484120125519222*pi,0.3297771952132402*pi) q[37];
U1q(0.879677163343534*pi,1.282679482628878*pi) q[38];
U1q(0.854674213388344*pi,1.7060748853206502*pi) q[39];
RZZ(0.5*pi) q[35],q[0];
RZZ(0.5*pi) q[29],q[1];
RZZ(0.5*pi) q[2],q[30];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[33],q[11];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[17],q[31];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[36],q[24];
RZZ(0.5*pi) q[39],q[25];
RZZ(0.5*pi) q[38],q[26];
RZZ(0.5*pi) q[32],q[34];
U1q(0.604479745996697*pi,0.7827614095038005*pi) q[0];
U1q(0.282105108707088*pi,0.8298859838555863*pi) q[1];
U1q(0.482983420895264*pi,0.40595228705882924*pi) q[2];
U1q(0.951061054195113*pi,0.9456328729478001*pi) q[3];
U1q(0.536963042642249*pi,1.5084630273760578*pi) q[4];
U1q(0.462680334295527*pi,0.38476933146707015*pi) q[5];
U1q(0.903791267317475*pi,0.06628865431583986*pi) q[6];
U1q(0.73102508705193*pi,1.6605634622511207*pi) q[7];
U1q(0.65634083198647*pi,1.07307097451362*pi) q[8];
U1q(0.62162417313102*pi,1.5784420901645797*pi) q[9];
U1q(0.559457808923398*pi,0.5317329253341985*pi) q[10];
U1q(0.0901719703877752*pi,1.6833305807508996*pi) q[11];
U1q(0.148186632216215*pi,0.6651749573736705*pi) q[12];
U1q(0.681337839749725*pi,1.0510002920079096*pi) q[13];
U1q(0.515398616922106*pi,1.4735106907380704*pi) q[14];
U1q(0.362111072994216*pi,1.1444548777946366*pi) q[15];
U1q(0.5545393002756*pi,0.5557590246845789*pi) q[16];
U1q(0.63726727913079*pi,0.5543543326907034*pi) q[17];
U1q(0.186837360988294*pi,0.7579851233557946*pi) q[18];
U1q(0.810100999972925*pi,0.4982284260423402*pi) q[19];
U1q(0.480501911853251*pi,1.6725916864882198*pi) q[20];
U1q(0.547440273797389*pi,0.03600261210559008*pi) q[21];
U1q(0.312272007913467*pi,1.8289380753130597*pi) q[22];
U1q(0.754847052336818*pi,0.24189884088526004*pi) q[23];
U1q(0.722048098497712*pi,1.3330516515093964*pi) q[24];
U1q(0.704529105163353*pi,1.1975098221191294*pi) q[25];
U1q(0.341314611753036*pi,1.9860314811531996*pi) q[26];
U1q(0.542799924512646*pi,0.042951319311680614*pi) q[27];
U1q(0.664129411545348*pi,1.9582618773860432*pi) q[28];
U1q(0.64498345696969*pi,0.6475649543114548*pi) q[29];
U1q(0.152649246824019*pi,0.09638540363205372*pi) q[30];
U1q(0.336846015095812*pi,1.15090997925122*pi) q[31];
U1q(0.26309074314939*pi,1.9655241274340813*pi) q[32];
U1q(0.511390011579503*pi,1.372011516718179*pi) q[33];
U1q(0.263971851546407*pi,0.8509806169361402*pi) q[34];
U1q(0.415982704991444*pi,0.0038051315657501306*pi) q[35];
U1q(0.35030562120512*pi,0.6694741041021341*pi) q[36];
U1q(0.508606998549325*pi,0.5805212184011097*pi) q[37];
U1q(0.71794475000878*pi,0.10682253926111995*pi) q[38];
U1q(0.451317584663305*pi,0.5600551146657899*pi) q[39];
RZZ(0.5*pi) q[19],q[0];
RZZ(0.5*pi) q[20],q[1];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[3],q[14];
RZZ(0.5*pi) q[5],q[30];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[25];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[23],q[9];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[22],q[37];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[31],q[27];
RZZ(0.5*pi) q[28],q[34];
RZZ(0.5*pi) q[36],q[29];
U1q(0.385663265314555*pi,1.9299410910324006*pi) q[0];
U1q(0.727693372918112*pi,1.298163586180527*pi) q[1];
U1q(0.750982635014143*pi,1.8031752793579*pi) q[2];
U1q(0.453294500637985*pi,0.2681844514824805*pi) q[3];
U1q(0.550048622564839*pi,1.9867692269299475*pi) q[4];
U1q(0.0537342906164735*pi,0.90213837074986*pi) q[5];
U1q(0.174412430535264*pi,0.9043449830779*pi) q[6];
U1q(0.354933260081893*pi,0.5875062871718999*pi) q[7];
U1q(0.975892562513058*pi,1.10250123273965*pi) q[8];
U1q(0.367369137289513*pi,1.27459008978507*pi) q[9];
U1q(0.180132349426096*pi,1.1840103599564689*pi) q[10];
U1q(0.460546006171586*pi,0.8766999130462008*pi) q[11];
U1q(0.189049553045887*pi,1.7533211515685405*pi) q[12];
U1q(0.79505239027239*pi,0.2510103434507194*pi) q[13];
U1q(0.732770820997548*pi,1.46158784316175*pi) q[14];
U1q(0.462628821362602*pi,1.1361945291070157*pi) q[15];
U1q(0.347257659501773*pi,0.7518989895669588*pi) q[16];
U1q(0.305761372713428*pi,0.8268157662861526*pi) q[17];
U1q(0.512544203443349*pi,0.41129267051209517*pi) q[18];
U1q(0.487347187314381*pi,1.4020494210976802*pi) q[19];
U1q(0.384922525326034*pi,0.4347711052792196*pi) q[20];
U1q(0.203658522925834*pi,1.5473507192739895*pi) q[21];
U1q(0.233926830822166*pi,0.6322192114919201*pi) q[22];
U1q(0.570669673681496*pi,0.6580626748814193*pi) q[23];
U1q(0.106206668576696*pi,0.5285679937363561*pi) q[24];
U1q(0.856119944491278*pi,1.1955778379558684*pi) q[25];
U1q(0.961234242942381*pi,0.0008151708208004038*pi) q[26];
U1q(0.40121838021266*pi,0.10472645519729973*pi) q[27];
U1q(0.525763078699284*pi,0.7566971024442033*pi) q[28];
U1q(0.441693746449231*pi,0.8923760756280359*pi) q[29];
U1q(0.624254221100975*pi,0.7468802837805626*pi) q[30];
U1q(0.333813806897849*pi,0.4431416365078702*pi) q[31];
U1q(0.135550649163275*pi,0.8887495929238209*pi) q[32];
U1q(0.559162014282241*pi,1.18138663234474*pi) q[33];
U1q(0.611854363359916*pi,0.31026319821273063*pi) q[34];
U1q(0.765992072218111*pi,1.6358405727916008*pi) q[35];
U1q(0.59736280099682*pi,1.4488105271137748*pi) q[36];
U1q(0.404793665955024*pi,0.30505552993349916*pi) q[37];
U1q(0.808574873178278*pi,0.2532250123258599*pi) q[38];
U1q(0.733053427908435*pi,1.3109241595652996*pi) q[39];
rz(2.9551757624406996*pi) q[0];
rz(2.6414743931614133*pi) q[1];
rz(1.7311863499555908*pi) q[2];
rz(1.2184827117261197*pi) q[3];
rz(3.2554767709117822*pi) q[4];
rz(1.5483657418903505*pi) q[5];
rz(2.4372608883499005*pi) q[6];
rz(3.7579484025725005*pi) q[7];
rz(0.5101644796120297*pi) q[8];
rz(1.3424688004515506*pi) q[9];
rz(0.24915563745637037*pi) q[10];
rz(1.0553016513029991*pi) q[11];
rz(1.0082160862173382*pi) q[12];
rz(1.0222474290160815*pi) q[13];
rz(0.45396079771767006*pi) q[14];
rz(2.079764699325093*pi) q[15];
rz(3.733221400751181*pi) q[16];
rz(3.586686596643988*pi) q[17];
rz(1.5979965614053047*pi) q[18];
rz(0.4405933157295401*pi) q[19];
rz(2.2043817891510002*pi) q[20];
rz(3.66178651811075*pi) q[21];
rz(2.8909451989561097*pi) q[22];
rz(2.8057923194894006*pi) q[23];
rz(1.1736036205383833*pi) q[24];
rz(1.7205549857409714*pi) q[25];
rz(0.9773157237593999*pi) q[26];
rz(3.8632625094299016*pi) q[27];
rz(2.4500470725513264*pi) q[28];
rz(2.209068021631456*pi) q[29];
rz(1.0153881239934677*pi) q[30];
rz(0.24874541372962966*pi) q[31];
rz(3.9328728675049778*pi) q[32];
rz(1.2618721952984*pi) q[33];
rz(2.3151758871086*pi) q[34];
rz(1.2255244901447906*pi) q[35];
rz(1.6193756611662362*pi) q[36];
rz(3.6360943870763993*pi) q[37];
rz(0.51340031498024*pi) q[38];
rz(0.32617372889639995*pi) q[39];
U1q(1.38566326531456*pi,1.885116853473136*pi) q[0];
U1q(1.72769337291811*pi,0.939637979341938*pi) q[1];
U1q(0.750982635014143*pi,0.534361629313515*pi) q[2];
U1q(1.45329450063799*pi,0.486667163208597*pi) q[3];
U1q(0.550048622564839*pi,0.24224599784173*pi) q[4];
U1q(1.05373429061647*pi,1.450504112640209*pi) q[5];
U1q(1.17441243053526*pi,0.341605871427741*pi) q[6];
U1q(0.354933260081893*pi,1.345454689744471*pi) q[7];
U1q(0.975892562513058*pi,0.612665712351679*pi) q[8];
U1q(1.36736913728951*pi,1.6170588902366139*pi) q[9];
U1q(0.180132349426096*pi,0.4331659974128399*pi) q[10];
U1q(1.46054600617159*pi,0.9320015643491499*pi) q[11];
U1q(1.18904955304589*pi,1.761537237785877*pi) q[12];
U1q(1.79505239027239*pi,0.273257772466827*pi) q[13];
U1q(0.732770820997548*pi,0.915548640879421*pi) q[14];
U1q(1.4626288213626*pi,0.215959228432126*pi) q[15];
U1q(1.34725765950177*pi,1.4851203903181411*pi) q[16];
U1q(0.305761372713428*pi,1.413502362930143*pi) q[17];
U1q(1.51254420344335*pi,1.009289231917391*pi) q[18];
U1q(1.48734718731438*pi,0.842642736827215*pi) q[19];
U1q(0.384922525326034*pi,1.639152894430198*pi) q[20];
U1q(0.203658522925834*pi,0.20913723738474*pi) q[21];
U1q(0.233926830822166*pi,0.523164410448033*pi) q[22];
U1q(0.570669673681496*pi,0.463854994370781*pi) q[23];
U1q(1.1062066685767*pi,0.7021716142747401*pi) q[24];
U1q(0.856119944491278*pi,1.9161328236967874*pi) q[25];
U1q(0.961234242942381*pi,1.9781308945801233*pi) q[26];
U1q(1.40121838021266*pi,0.96798896462717*pi) q[27];
U1q(1.52576307869928*pi,0.206744174995579*pi) q[28];
U1q(0.441693746449231*pi,0.101444097259497*pi) q[29];
U1q(1.62425422110098*pi,0.762268407774032*pi) q[30];
U1q(0.333813806897849*pi,1.691887050237501*pi) q[31];
U1q(0.135550649163275*pi,1.821622460428811*pi) q[32];
U1q(0.559162014282241*pi,1.443258827643148*pi) q[33];
U1q(0.611854363359916*pi,1.625439085321364*pi) q[34];
U1q(0.765992072218111*pi,1.861365062936394*pi) q[35];
U1q(1.59736280099682*pi,0.0681861882800092*pi) q[36];
U1q(1.40479366595502*pi,0.941149917009959*pi) q[37];
U1q(0.808574873178278*pi,1.766625327306099*pi) q[38];
U1q(3.733053427908435*pi,0.637097888461705*pi) q[39];
RZZ(0.5*pi) q[19],q[0];
RZZ(0.5*pi) q[20],q[1];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[3],q[14];
RZZ(0.5*pi) q[5],q[30];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[25];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[23],q[9];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[22],q[37];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[31],q[27];
RZZ(0.5*pi) q[28],q[34];
RZZ(0.5*pi) q[36],q[29];
U1q(1.6044797459967*pi,0.032296535001786975*pi) q[0];
U1q(3.717894891292912*pi,0.40791558166687636*pi) q[1];
U1q(0.482983420895264*pi,0.13713863701442008*pi) q[2];
U1q(1.95106105419511*pi,1.8092187417432724*pi) q[3];
U1q(0.536963042642249*pi,0.76393979828784*pi) q[4];
U1q(1.46268033429553*pi,0.9678731519229995*pi) q[5];
U1q(1.90379126731748*pi,0.1796622001897915*pi) q[6];
U1q(1.73102508705193*pi,1.41851186482365*pi) q[7];
U1q(1.65634083198647*pi,1.5832354541256501*pi) q[8];
U1q(1.62162417313102*pi,1.3132068898571008*pi) q[9];
U1q(0.559457808923398*pi,0.78088856279057*pi) q[10];
U1q(3.0901719703877752*pi,0.1253708966444389*pi) q[11];
U1q(3.851813367783785*pi,1.849683431980746*pi) q[12];
U1q(3.681337839749725*pi,1.4732678239096555*pi) q[13];
U1q(1.51539861692211*pi,0.92747148845574*pi) q[14];
U1q(3.637888927005784*pi,1.2076988797445216*pi) q[15];
U1q(1.5545393002756*pi,1.6812603552005194*pi) q[16];
U1q(1.63726727913079*pi,1.141040929334691*pi) q[17];
U1q(3.186837360988294*pi,1.6625967790736658*pi) q[18];
U1q(1.81010099997292*pi,0.7464637318825478*pi) q[19];
U1q(1.48050191185325*pi,0.87697347563919*pi) q[20];
U1q(0.547440273797389*pi,1.697789130216343*pi) q[21];
U1q(1.31227200791347*pi,1.719883274269168*pi) q[22];
U1q(1.75484705233682*pi,0.047691160374619956*pi) q[23];
U1q(1.72204809849771*pi,0.897687956501696*pi) q[24];
U1q(1.70452910516335*pi,0.9180648078600502*pi) q[25];
U1q(1.34131461175304*pi,1.9633472049125502*pi) q[26];
U1q(3.457200075487354*pi,0.029764100512804648*pi) q[27];
U1q(1.66412941154535*pi,1.005179400053746*pi) q[28];
U1q(1.64498345696969*pi,1.8566329759429059*pi) q[29];
U1q(3.847350753175979*pi,1.4127632879225356*pi) q[30];
U1q(0.336846015095812*pi,0.39965539298085995*pi) q[31];
U1q(1.26309074314939*pi,0.89839699493907*pi) q[32];
U1q(0.511390011579503*pi,1.6338837120165897*pi) q[33];
U1q(1.26397185154641*pi,0.1661565040447801*pi) q[34];
U1q(0.415982704991444*pi,1.229329621710538*pi) q[35];
U1q(1.35030562120512*pi,0.8475226112916505*pi) q[36];
U1q(3.491393001450675*pi,1.6656842285423892*pi) q[37];
U1q(1.71794475000878*pi,1.6202228542413502*pi) q[38];
U1q(3.548682415336695*pi,1.3879669333612183*pi) q[39];
RZZ(0.5*pi) q[35],q[0];
RZZ(0.5*pi) q[29],q[1];
RZZ(0.5*pi) q[2],q[30];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[33],q[11];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[17],q[31];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[36],q[24];
RZZ(0.5*pi) q[39],q[25];
RZZ(0.5*pi) q[38],q[26];
RZZ(0.5*pi) q[32],q[34];
U1q(1.32926919663023*pi,0.8081847926076167*pi) q[0];
U1q(3.613154777850343*pi,0.5125302767366663*pi) q[1];
U1q(0.404886039227992*pi,0.10578043425640993*pi) q[2];
U1q(1.87579122319267*pi,1.9118451754841954*pi) q[3];
U1q(1.35166592463121*pi,1.1361559365153102*pi) q[4];
U1q(1.49979826804942*pi,1.4181227977646094*pi) q[5];
U1q(3.16630416900981*pi,1.7445566988419303*pi) q[6];
U1q(3.892530082480845*pi,1.600901875688284*pi) q[7];
U1q(3.420441939617503*pi,0.8431913577543382*pi) q[8];
U1q(1.40093385965475*pi,0.6274380146278866*pi) q[9];
U1q(1.85092364872244*pi,1.5567449166125602*pi) q[10];
U1q(0.823582252813157*pi,1.4337569065426687*pi) q[11];
U1q(3.428919964858248*pi,1.5340463786223526*pi) q[12];
U1q(1.31796411893039*pi,0.5591749297288984*pi) q[13];
U1q(3.529385378387856*pi,0.5674619048024216*pi) q[14];
U1q(1.81621901293964*pi,1.4468236673290842*pi) q[15];
U1q(1.60415398652832*pi,1.9613474884738884*pi) q[16];
U1q(3.675048467460801*pi,1.0946030779708638*pi) q[17];
U1q(0.203343901760457*pi,1.2499030435981258*pi) q[18];
U1q(0.840990604587761*pi,0.7153737319826577*pi) q[19];
U1q(3.85681002577486*pi,1.723551553588913*pi) q[20];
U1q(0.108770212331605*pi,1.3941700424431902*pi) q[21];
U1q(1.69132561874423*pi,0.8223174608630595*pi) q[22];
U1q(3.7470296274957002*pi,1.6036656836559091*pi) q[23];
U1q(0.494893156002058*pi,1.4667666182729961*pi) q[24];
U1q(3.831091154935456*pi,1.4113028299297534*pi) q[25];
U1q(1.59203813463676*pi,1.0777135624726037*pi) q[26];
U1q(1.23026950048429*pi,0.48153247748176464*pi) q[27];
U1q(0.290202419930001*pi,0.32522219437890887*pi) q[28];
U1q(1.76814108463124*pi,1.5159974833957646*pi) q[29];
U1q(3.306380651573854*pi,1.2199334311852177*pi) q[30];
U1q(0.308792784761276*pi,1.659865828455477*pi) q[31];
U1q(1.12217308086275*pi,1.7268528185406846*pi) q[32];
U1q(1.34385315212516*pi,0.7889982709523098*pi) q[33];
U1q(3.460850801569013*pi,1.9348275766941763*pi) q[34];
U1q(0.621811200448577*pi,1.683084341344767*pi) q[35];
U1q(0.708998382473311*pi,0.5370696248691813*pi) q[36];
U1q(3.484120125519222*pi,1.9164282517302564*pi) q[37];
U1q(3.879677163343535*pi,1.4443659108735911*pi) q[38];
U1q(1.85467421338834*pi,0.2419471627063578*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[32],q[4];
RZZ(0.5*pi) q[5],q[20];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[16],q[39];
RZZ(0.5*pi) q[21],q[26];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[28],q[24];
RZZ(0.5*pi) q[38],q[30];
RZZ(0.5*pi) q[34],q[33];
RZZ(0.5*pi) q[37],q[35];
U1q(3.4628917917934032*pi,0.45233261087275967*pi) q[0];
U1q(3.6532035434132712*pi,0.7802657083337364*pi) q[1];
U1q(1.6405894140468*pi,0.36520293155074013*pi) q[2];
U1q(3.280553787811254*pi,0.6222397144576908*pi) q[3];
U1q(1.7456240947063*pi,1.4689602094193486*pi) q[4];
U1q(3.276567082903245*pi,0.3310909779400788*pi) q[5];
U1q(1.75604269633451*pi,1.960377057194834*pi) q[6];
U1q(3.502762596089049*pi,1.0335670619211041*pi) q[7];
U1q(3.263863244440329*pi,0.7396482774901452*pi) q[8];
U1q(3.542158580570906*pi,0.2157400469938695*pi) q[9];
U1q(3.234592026746294*pi,1.425785648855666*pi) q[10];
U1q(0.164526113962273*pi,1.4687514482471586*pi) q[11];
U1q(0.395719391307192*pi,0.3543938872248025*pi) q[12];
U1q(3.59334126200344*pi,0.12085531564670982*pi) q[13];
U1q(3.389465421415268*pi,0.9254388775593216*pi) q[14];
U1q(0.736283163760351*pi,0.9614183539124239*pi) q[15];
U1q(3.887236566016357*pi,1.695770287631753*pi) q[16];
U1q(1.78784060643183*pi,1.7529754716947026*pi) q[17];
U1q(0.656164000177277*pi,0.7276169947664659*pi) q[18];
U1q(1.54430318327588*pi,0.7593412246044128*pi) q[19];
U1q(3.750568109912623*pi,0.041756082787722915*pi) q[20];
U1q(1.08934059763833*pi,0.31133294381840004*pi) q[21];
U1q(0.533943692657556*pi,0.3333959046928594*pi) q[22];
U1q(1.25186417165169*pi,0.1583834789298164*pi) q[23];
U1q(0.41036894761538*pi,1.5168454681415762*pi) q[24];
U1q(1.15607651178472*pi,0.446154287061197*pi) q[25];
U1q(0.468797884526534*pi,1.2132059505981232*pi) q[26];
U1q(1.07464529971049*pi,0.38501656487948477*pi) q[27];
U1q(1.44709517709325*pi,0.14798730719652697*pi) q[28];
U1q(0.408283844210586*pi,1.8020700550353346*pi) q[29];
U1q(1.37335719609945*pi,1.92875337131206*pi) q[30];
U1q(1.3508648134657*pi,0.43694247121427*pi) q[31];
U1q(1.88652508582416*pi,1.3377049865223944*pi) q[32];
U1q(3.553491960483903*pi,0.6326916235063189*pi) q[33];
U1q(1.42555788007431*pi,0.6833738179633553*pi) q[34];
U1q(3.250176438631044*pi,1.66011035276364*pi) q[35];
U1q(1.8038437072044*pi,0.0887720659347615*pi) q[36];
U1q(0.381771064396081*pi,1.6358904265558825*pi) q[37];
U1q(1.82361712461489*pi,1.278581200966821*pi) q[38];
U1q(0.490021828855255*pi,0.5078818185568077*pi) q[39];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[30],q[1];
RZZ(0.5*pi) q[2],q[9];
RZZ(0.5*pi) q[3],q[32];
RZZ(0.5*pi) q[15],q[4];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[8],q[34];
RZZ(0.5*pi) q[24],q[10];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[13],q[29];
RZZ(0.5*pi) q[14],q[23];
RZZ(0.5*pi) q[16],q[35];
RZZ(0.5*pi) q[17],q[36];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[19],q[39];
RZZ(0.5*pi) q[21],q[22];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[28],q[37];
U1q(1.78039452171598*pi,0.7854145835868138*pi) q[0];
U1q(1.34451725871686*pi,0.2308363026962475*pi) q[1];
U1q(1.37503906077521*pi,0.18543423928932246*pi) q[2];
U1q(3.075983885105893*pi,0.3301095409305512*pi) q[3];
U1q(0.307528572073703*pi,1.5412266110145785*pi) q[4];
U1q(1.21027194313419*pi,0.22863255616402212*pi) q[5];
U1q(0.748988954002769*pi,0.8625057329978838*pi) q[6];
U1q(1.32213835087057*pi,1.835490742467731*pi) q[7];
U1q(1.50937631904071*pi,1.1763812446235602*pi) q[8];
U1q(1.5843198723596*pi,1.5539954157396139*pi) q[9];
U1q(3.214242149529192*pi,1.207454208787774*pi) q[10];
U1q(0.34035762450952*pi,0.5432654218097888*pi) q[11];
U1q(0.569737301900453*pi,0.27839776113383286*pi) q[12];
U1q(3.631113760635354*pi,0.2883817672107434*pi) q[13];
U1q(1.77895611806152*pi,1.297367027208411*pi) q[14];
U1q(0.37470399956954*pi,0.33993172338922406*pi) q[15];
U1q(3.63884347368798*pi,1.2890671996931098*pi) q[16];
U1q(1.65556627185292*pi,0.4227362763633331*pi) q[17];
U1q(0.739072395045494*pi,0.1291828012172349*pi) q[18];
U1q(1.76029638521546*pi,0.13221755764276769*pi) q[19];
U1q(1.33313595930972*pi,0.5146513002013768*pi) q[20];
U1q(3.121523228222043*pi,1.0936845240224553*pi) q[21];
U1q(0.617682345071073*pi,0.21818371146418958*pi) q[22];
U1q(0.0612598207646955*pi,0.8318871855200864*pi) q[23];
U1q(0.380509692543887*pi,0.4486408032424962*pi) q[24];
U1q(0.622188343607556*pi,0.4246386247133671*pi) q[25];
U1q(0.350690416354307*pi,1.8958563261099828*pi) q[26];
U1q(1.67853487249723*pi,0.3308702045290408*pi) q[27];
U1q(1.52081424828409*pi,1.0566635276085816*pi) q[28];
U1q(0.27275749056176*pi,0.6903653469688944*pi) q[29];
U1q(0.358137212301125*pi,0.80899903664467*pi) q[30];
U1q(1.47320382330082*pi,0.6813497041496799*pi) q[31];
U1q(1.24917098029235*pi,0.9243406469725546*pi) q[32];
U1q(1.47626070134231*pi,0.39428422092351845*pi) q[33];
U1q(0.179596702820872*pi,0.990900126793675*pi) q[34];
U1q(1.64260157543576*pi,0.7033012605833191*pi) q[35];
U1q(1.86851605662078*pi,0.7936404502349927*pi) q[36];
U1q(0.76189229260038*pi,0.10340001697839218*pi) q[37];
U1q(1.69911272759021*pi,0.32042661218835056*pi) q[38];
U1q(0.163213063220668*pi,0.2819310193245679*pi) q[39];
rz(1.2145854164131862*pi) q[0];
rz(3.7691636973037523*pi) q[1];
rz(3.8145657607106775*pi) q[2];
rz(3.6698904590694488*pi) q[3];
rz(0.4587733889854215*pi) q[4];
rz(1.7713674438359779*pi) q[5];
rz(1.1374942670021162*pi) q[6];
rz(0.16450925753226908*pi) q[7];
rz(0.8236187553764398*pi) q[8];
rz(0.44600458426038614*pi) q[9];
rz(2.792545791212226*pi) q[10];
rz(1.4567345781902112*pi) q[11];
rz(3.721602238866167*pi) q[12];
rz(1.7116182327892566*pi) q[13];
rz(0.7026329727915891*pi) q[14];
rz(1.660068276610776*pi) q[15];
rz(2.71093280030689*pi) q[16];
rz(1.577263723636667*pi) q[17];
rz(1.870817198782765*pi) q[18];
rz(3.8677824423572322*pi) q[19];
rz(1.4853486997986232*pi) q[20];
rz(0.9063154759775447*pi) q[21];
rz(3.7818162885358104*pi) q[22];
rz(1.1681128144799136*pi) q[23];
rz(3.5513591967575038*pi) q[24];
rz(1.575361375286633*pi) q[25];
rz(2.104143673890017*pi) q[26];
rz(3.669129795470959*pi) q[27];
rz(0.9433364723914184*pi) q[28];
rz(3.3096346530311056*pi) q[29];
rz(1.19100096335533*pi) q[30];
rz(1.3186502958503201*pi) q[31];
rz(3.0756593530274454*pi) q[32];
rz(1.6057157790764816*pi) q[33];
rz(3.009099873206325*pi) q[34];
rz(3.296698739416681*pi) q[35];
rz(3.2063595497650073*pi) q[36];
rz(3.896599983021608*pi) q[37];
rz(3.6795733878116494*pi) q[38];
rz(3.718068980675432*pi) q[39];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];
measure q[28] -> c[28];
measure q[29] -> c[29];
measure q[30] -> c[30];
measure q[31] -> c[31];
measure q[32] -> c[32];
measure q[33] -> c[33];
measure q[34] -> c[34];
measure q[35] -> c[35];
measure q[36] -> c[36];
measure q[37] -> c[37];
measure q[38] -> c[38];
measure q[39] -> c[39];