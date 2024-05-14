OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(3.449683125097251*pi,0.9848810115262369*pi) q[0];
U1q(0.382364452128992*pi,1.01930298047018*pi) q[1];
U1q(0.397086920330642*pi,1.871183301997087*pi) q[2];
U1q(1.42261436854895*pi,0.2686034934904782*pi) q[3];
U1q(1.59447916797642*pi,0.5177300933605782*pi) q[4];
U1q(1.68777872322988*pi,1.8188627227850556*pi) q[5];
U1q(0.59812932334777*pi,0.135712986020261*pi) q[6];
U1q(0.525706833659193*pi,0.061053441886515*pi) q[7];
U1q(1.62451806188652*pi,1.4287847399116065*pi) q[8];
U1q(0.436846030532685*pi,1.42906997414546*pi) q[9];
U1q(1.75940017964911*pi,1.0334622559286388*pi) q[10];
U1q(1.39367547885165*pi,0.34279115211990985*pi) q[11];
U1q(0.300328381234662*pi,0.00491093116781749*pi) q[12];
U1q(1.5563258351536*pi,1.0942479052293437*pi) q[13];
U1q(1.44146935626417*pi,0.48625673500563765*pi) q[14];
U1q(0.607065016006823*pi,0.0979745452399339*pi) q[15];
U1q(1.63725102109939*pi,0.43591740570317633*pi) q[16];
U1q(0.479617792642381*pi,0.596198559160785*pi) q[17];
U1q(0.386244099143304*pi,1.474479217452427*pi) q[18];
U1q(0.731898911762354*pi,0.722643250882519*pi) q[19];
U1q(0.786234134819382*pi,0.454248044130163*pi) q[20];
U1q(1.06194866353367*pi,1.2020757573315697*pi) q[21];
U1q(0.699488917384372*pi,1.389058743999785*pi) q[22];
U1q(0.737936847364053*pi,1.25811593385614*pi) q[23];
U1q(0.65789297860459*pi,1.9010002380477866*pi) q[24];
U1q(1.28299964201865*pi,1.5106969515407214*pi) q[25];
U1q(0.375143705295409*pi,0.0360351838898582*pi) q[26];
U1q(0.63185373607045*pi,1.38768717587441*pi) q[27];
U1q(1.48401493189706*pi,1.3461759224251442*pi) q[28];
U1q(0.736353268501761*pi,1.709951415966427*pi) q[29];
U1q(1.89489475314653*pi,0.47975830528073615*pi) q[30];
U1q(1.74738317671457*pi,0.7266930589274972*pi) q[31];
U1q(0.768152621713417*pi,1.870648874665821*pi) q[32];
U1q(0.783113395232281*pi,0.581360900123703*pi) q[33];
U1q(0.136615724200892*pi,0.0130042887883746*pi) q[34];
U1q(1.63834888302398*pi,1.366598478395498*pi) q[35];
U1q(0.47340393027828*pi,0.0333025936596947*pi) q[36];
U1q(1.81735151019318*pi,0.1402457974740193*pi) q[37];
U1q(0.576319096791119*pi,0.526152350543463*pi) q[38];
U1q(0.38337227779845*pi,0.842575942584708*pi) q[39];
RZZ(0.5*pi) q[0],q[36];
RZZ(0.5*pi) q[1],q[26];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[15];
RZZ(0.5*pi) q[4],q[38];
RZZ(0.5*pi) q[9],q[6];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[8],q[21];
RZZ(0.5*pi) q[28],q[10];
RZZ(0.5*pi) q[24],q[11];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[14],q[13];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[33],q[17];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[31],q[35];
RZZ(0.5*pi) q[37],q[39];
U1q(0.702769109353906*pi,0.08942735504137689*pi) q[0];
U1q(0.148188138767072*pi,1.328838846485316*pi) q[1];
U1q(0.319237359065678*pi,0.4447213174443001*pi) q[2];
U1q(0.562724332638666*pi,1.5725090290438581*pi) q[3];
U1q(0.156098765414584*pi,0.17142479829400825*pi) q[4];
U1q(0.318330706116185*pi,1.4241677152303756*pi) q[5];
U1q(0.726495223067824*pi,0.12851690397288995*pi) q[6];
U1q(0.62871331128577*pi,1.7441738951094*pi) q[7];
U1q(0.0264673762582222*pi,0.37014709009794666*pi) q[8];
U1q(0.619205974037159*pi,0.1998226946588999*pi) q[9];
U1q(0.81205766876093*pi,1.2451129387747089*pi) q[10];
U1q(0.573447237333904*pi,0.5907349749320798*pi) q[11];
U1q(0.507352245292396*pi,1.1208047442131601*pi) q[12];
U1q(0.17725297893801*pi,0.018260940268773673*pi) q[13];
U1q(0.382114435720488*pi,0.6003590062818076*pi) q[14];
U1q(0.213513682553843*pi,1.3134220785531299*pi) q[15];
U1q(0.0991920555154996*pi,0.2848916478882364*pi) q[16];
U1q(0.343044064907713*pi,0.5029617055039699*pi) q[17];
U1q(0.656217826101*pi,0.33301485003753983*pi) q[18];
U1q(0.438336674187113*pi,1.9906940575965701*pi) q[19];
U1q(0.639493215132442*pi,1.8578066786909702*pi) q[20];
U1q(0.825624742953929*pi,0.4736454929919396*pi) q[21];
U1q(0.501169949721164*pi,1.05292322906506*pi) q[22];
U1q(0.694467626923437*pi,1.60024891928142*pi) q[23];
U1q(0.097933364641084*pi,1.10910973770113*pi) q[24];
U1q(0.455202454503498*pi,0.24964800417749178*pi) q[25];
U1q(0.52788319401891*pi,0.7356087016113202*pi) q[26];
U1q(0.66921317258594*pi,1.269802523869638*pi) q[27];
U1q(0.505925250543034*pi,0.8869412847384242*pi) q[28];
U1q(0.761936287246064*pi,0.14299347116920003*pi) q[29];
U1q(0.148179618900661*pi,1.3621214122528262*pi) q[30];
U1q(0.232696894301172*pi,1.2169584083292673*pi) q[31];
U1q(0.407196105821579*pi,1.5472873952748798*pi) q[32];
U1q(0.409451762948382*pi,0.82853944208572*pi) q[33];
U1q(0.257969996563115*pi,1.22712522846741*pi) q[34];
U1q(0.672811490435971*pi,1.739753551557106*pi) q[35];
U1q(0.643863230851743*pi,0.1473372420838599*pi) q[36];
U1q(0.327872188454165*pi,0.12067944301733924*pi) q[37];
U1q(0.314695502229745*pi,1.5883120958547599*pi) q[38];
U1q(0.923102396627794*pi,0.6699955250976899*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[4],q[26];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[21],q[6];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[8],q[36];
RZZ(0.5*pi) q[9],q[34];
RZZ(0.5*pi) q[10],q[22];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[13],q[37];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[25],q[17];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[30],q[27];
RZZ(0.5*pi) q[28],q[35];
RZZ(0.5*pi) q[32],q[33];
U1q(0.549784065410506*pi,0.8996414583335168*pi) q[0];
U1q(0.588442650006978*pi,0.7922636118468*pi) q[1];
U1q(0.357332815416553*pi,0.5412230119351698*pi) q[2];
U1q(0.607406841713066*pi,1.054163249877508*pi) q[3];
U1q(0.0270219771311147*pi,1.9613102617968883*pi) q[4];
U1q(0.737354190395379*pi,0.0856930618514955*pi) q[5];
U1q(0.702451620578378*pi,0.025617591489880187*pi) q[6];
U1q(0.338607560182068*pi,0.7478718385012302*pi) q[7];
U1q(0.717695696106128*pi,1.7642407924568762*pi) q[8];
U1q(0.480786043056467*pi,0.91195402464195*pi) q[9];
U1q(0.385801919096872*pi,1.703875091675049*pi) q[10];
U1q(0.0706038815780688*pi,1.2515412115442697*pi) q[11];
U1q(0.418367167361903*pi,0.9015353418007299*pi) q[12];
U1q(0.645918240174644*pi,1.7198233867431636*pi) q[13];
U1q(0.594400312942993*pi,1.6999075537027677*pi) q[14];
U1q(0.258868526224039*pi,1.76143253006299*pi) q[15];
U1q(0.504688003984608*pi,1.9533349618765667*pi) q[16];
U1q(0.59538940981585*pi,1.95989927869124*pi) q[17];
U1q(0.257621401940699*pi,0.26071965314629963*pi) q[18];
U1q(0.405099629194209*pi,1.3548263929007103*pi) q[19];
U1q(0.762001029878919*pi,0.07736303234578035*pi) q[20];
U1q(0.609653937203965*pi,0.5935567291891495*pi) q[21];
U1q(0.385845262332028*pi,1.83048462289217*pi) q[22];
U1q(0.338159995404855*pi,1.3545174881607398*pi) q[23];
U1q(0.208311114107046*pi,0.43081162355244995*pi) q[24];
U1q(0.875622365259298*pi,1.3293891988688413*pi) q[25];
U1q(0.302963427037614*pi,0.9251818758269001*pi) q[26];
U1q(0.468686125080435*pi,1.7939866233876502*pi) q[27];
U1q(0.643559926483582*pi,0.8399731222904538*pi) q[28];
U1q(0.138911319285078*pi,0.66261137896434*pi) q[29];
U1q(0.471447278753266*pi,1.9804234800868565*pi) q[30];
U1q(0.526902264319072*pi,1.313235707145858*pi) q[31];
U1q(0.181600627372754*pi,0.3659514678583702*pi) q[32];
U1q(0.211937791685751*pi,1.42321098331188*pi) q[33];
U1q(0.21535152565711*pi,1.10306773882564*pi) q[34];
U1q(0.525998635433907*pi,0.25871701963473814*pi) q[35];
U1q(0.293758485163122*pi,1.9430392877322804*pi) q[36];
U1q(0.230014181967507*pi,0.21196073101707924*pi) q[37];
U1q(0.559454443305219*pi,1.67411718040902*pi) q[38];
U1q(0.45724151226566*pi,1.79094474382884*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[20],q[1];
RZZ(0.5*pi) q[2],q[22];
RZZ(0.5*pi) q[3],q[21];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[23],q[10];
RZZ(0.5*pi) q[11],q[18];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[24],q[13];
RZZ(0.5*pi) q[14],q[37];
RZZ(0.5*pi) q[16],q[15];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[25],q[31];
RZZ(0.5*pi) q[39],q[27];
RZZ(0.5*pi) q[29],q[32];
RZZ(0.5*pi) q[34],q[30];
RZZ(0.5*pi) q[35],q[36];
U1q(0.933435905798736*pi,1.0855113776384364*pi) q[0];
U1q(0.5219735013556*pi,1.40850372943091*pi) q[1];
U1q(0.132052250287378*pi,1.0150466798971696*pi) q[2];
U1q(0.486238810234766*pi,1.4688637420727382*pi) q[3];
U1q(0.299058428317944*pi,1.1776507757447083*pi) q[4];
U1q(0.220285916129051*pi,1.0526623024772563*pi) q[5];
U1q(0.759093140657792*pi,0.8020917712871496*pi) q[6];
U1q(0.0783673411921988*pi,1.33237849195684*pi) q[7];
U1q(0.528344023273895*pi,0.07461584390504683*pi) q[8];
U1q(0.337915174557826*pi,1.18701663681174*pi) q[9];
U1q(0.928166738682204*pi,1.997976063846659*pi) q[10];
U1q(0.155469443245878*pi,0.57106668231933*pi) q[11];
U1q(0.529141889172377*pi,1.6660326785384107*pi) q[12];
U1q(0.681556854866279*pi,1.5689037725809332*pi) q[13];
U1q(0.169526330167391*pi,0.5203585995653173*pi) q[14];
U1q(0.216960603071725*pi,1.7471206847610699*pi) q[15];
U1q(0.0910197070736975*pi,0.15297714011907715*pi) q[16];
U1q(0.122923331611362*pi,0.01972925420080962*pi) q[17];
U1q(0.386285738137542*pi,0.15963537877905942*pi) q[18];
U1q(0.367321757078715*pi,0.9531109269234097*pi) q[19];
U1q(0.105418416485896*pi,1.0779841511433599*pi) q[20];
U1q(0.771578227878079*pi,0.5005585512893198*pi) q[21];
U1q(0.407869293381459*pi,0.2658530259205998*pi) q[22];
U1q(0.328228175701516*pi,0.6554508322876096*pi) q[23];
U1q(0.790939503597061*pi,1.4414102383916596*pi) q[24];
U1q(0.677776468256141*pi,0.5976060441531317*pi) q[25];
U1q(0.536863145184213*pi,0.9437284900374197*pi) q[26];
U1q(0.715896320932771*pi,0.7120396933198503*pi) q[27];
U1q(0.677485918859165*pi,0.8236968348931137*pi) q[28];
U1q(0.276704147197584*pi,0.9477539468071399*pi) q[29];
U1q(0.712858543407099*pi,1.2158456031869864*pi) q[30];
U1q(0.725070637924527*pi,0.35704586650129766*pi) q[31];
U1q(0.187688990711181*pi,1.56967533779006*pi) q[32];
U1q(0.684235037668083*pi,0.2929744075920304*pi) q[33];
U1q(0.250842887650374*pi,1.4320963761855996*pi) q[34];
U1q(0.476933044956989*pi,1.6125937793275575*pi) q[35];
U1q(0.571489985548581*pi,1.8987337773831108*pi) q[36];
U1q(0.849123026684764*pi,1.685155078140049*pi) q[37];
U1q(0.331554128453989*pi,0.09227587229795997*pi) q[38];
U1q(0.0341255378958414*pi,1.22745742333879*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[37];
RZZ(0.5*pi) q[2],q[34];
RZZ(0.5*pi) q[3],q[33];
RZZ(0.5*pi) q[4],q[25];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[13],q[18];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[16],q[19];
RZZ(0.5*pi) q[29],q[17];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[23],q[31];
RZZ(0.5*pi) q[38],q[26];
RZZ(0.5*pi) q[28],q[39];
U1q(0.25253515159907*pi,0.4202015591996382*pi) q[0];
U1q(0.498862702765466*pi,0.6759768302754603*pi) q[1];
U1q(0.130587338813907*pi,1.0656912226778097*pi) q[2];
U1q(0.859203363141481*pi,0.9071982483374583*pi) q[3];
U1q(0.293782174469473*pi,1.560257142891679*pi) q[4];
U1q(0.137764045457673*pi,0.7898523078200856*pi) q[5];
U1q(0.568074688246877*pi,1.2549868650877096*pi) q[6];
U1q(0.420369962767108*pi,1.4581465315262605*pi) q[7];
U1q(0.138102427715134*pi,0.0802748735969967*pi) q[8];
U1q(0.804534529403414*pi,0.68016357117307*pi) q[9];
U1q(0.334499649038932*pi,0.8963408969299387*pi) q[10];
U1q(0.220477414997667*pi,1.3371508434823092*pi) q[11];
U1q(0.0291940610921111*pi,1.1780706157825005*pi) q[12];
U1q(0.369524541285819*pi,0.8651886497569024*pi) q[13];
U1q(0.653297903546082*pi,1.2166376865049973*pi) q[14];
U1q(0.475876291242092*pi,1.2482872882871607*pi) q[15];
U1q(0.3878152287293*pi,1.139485816763285*pi) q[16];
U1q(0.111483176330086*pi,0.5558633201437697*pi) q[17];
U1q(0.73866997373556*pi,1.6511001570028991*pi) q[18];
U1q(0.0702495264166125*pi,0.8709652662392298*pi) q[19];
U1q(0.534788067113009*pi,0.3083025800989603*pi) q[20];
U1q(0.669124569556867*pi,1.258451897448591*pi) q[21];
U1q(0.797507816141234*pi,1.6028623016633006*pi) q[22];
U1q(0.279825186938063*pi,0.3269694092197408*pi) q[23];
U1q(0.092897340037183*pi,0.7375325008250702*pi) q[24];
U1q(0.117041572566555*pi,0.8548429539443205*pi) q[25];
U1q(0.204940762112535*pi,1.9921203039688002*pi) q[26];
U1q(0.854382249906369*pi,1.8350106996518196*pi) q[27];
U1q(0.462339122450147*pi,1.0832744855933143*pi) q[28];
U1q(0.293247497182333*pi,1.5882866336007506*pi) q[29];
U1q(0.423037841387454*pi,0.043050552621835436*pi) q[30];
U1q(0.353133564506943*pi,0.6953555772718971*pi) q[31];
U1q(0.348154692542138*pi,0.18087464126555997*pi) q[32];
U1q(0.449686364930924*pi,0.3370061314316697*pi) q[33];
U1q(0.0511557925749152*pi,1.3569772705901997*pi) q[34];
U1q(0.642190034735442*pi,0.9880539166510278*pi) q[35];
U1q(0.445363106309734*pi,0.3689813984171*pi) q[36];
U1q(0.639078712613505*pi,0.0077411992602201*pi) q[37];
U1q(0.784187844119591*pi,1.9218984166121995*pi) q[38];
U1q(0.0320833848217934*pi,0.9550622145047001*pi) q[39];
rz(1.685291223600263*pi) q[0];
rz(3.3543736082837396*pi) q[1];
rz(1.9536277547725796*pi) q[2];
rz(0.47221614288622327*pi) q[3];
rz(2.0748895004870214*pi) q[4];
rz(1.235592047733494*pi) q[5];
rz(2.5608113587169203*pi) q[6];
rz(1.9208966252440405*pi) q[7];
rz(1.4313808416653835*pi) q[8];
rz(0.7570648490980094*pi) q[9];
rz(0.41498147840806077*pi) q[10];
rz(2.449385607724791*pi) q[11];
rz(3.7627447163536996*pi) q[12];
rz(1.733350307800146*pi) q[13];
rz(3.994180243829703*pi) q[14];
rz(1.2047326457525607*pi) q[15];
rz(3.834125091339893*pi) q[16];
rz(2.2593046492987003*pi) q[17];
rz(3.5517616524134006*pi) q[18];
rz(1.5442164418034494*pi) q[19];
rz(1.3507056956410395*pi) q[20];
rz(3.9034815418397315*pi) q[21];
rz(0.6109410175702905*pi) q[22];
rz(2.1573296767044*pi) q[23];
rz(1.4309620479758003*pi) q[24];
rz(1.8703199841349782*pi) q[25];
rz(2.3059398680706007*pi) q[26];
rz(1.8261252870702096*pi) q[27];
rz(2.4743631038652154*pi) q[28];
rz(1.7791543695884204*pi) q[29];
rz(1.3248517172975145*pi) q[30];
rz(3.1641806351841026*pi) q[31];
rz(0.39161824368589926*pi) q[32];
rz(1.6560681785699494*pi) q[33];
rz(3.7890950942468997*pi) q[34];
rz(0.41014138455389215*pi) q[35];
rz(3.616486760027099*pi) q[36];
rz(3.0269412266160813*pi) q[37];
rz(2.9196395487033993*pi) q[38];
rz(2.7303078401583*pi) q[39];
U1q(1.25253515159907*pi,1.10549278279995*pi) q[0];
U1q(1.49886270276547*pi,1.0303504385592*pi) q[1];
U1q(1.13058733881391*pi,0.0193189774503864*pi) q[2];
U1q(1.85920336314148*pi,0.379414391223687*pi) q[3];
U1q(0.293782174469473*pi,0.6351466433786801*pi) q[4];
U1q(0.137764045457673*pi,1.025444355553574*pi) q[5];
U1q(0.568074688246877*pi,0.81579822380463*pi) q[6];
U1q(0.420369962767108*pi,0.37904315677029*pi) q[7];
U1q(0.138102427715134*pi,0.511655715262377*pi) q[8];
U1q(0.804534529403414*pi,0.437228420271077*pi) q[9];
U1q(1.33449964903893*pi,0.311322375338004*pi) q[10];
U1q(0.220477414997667*pi,0.78653645120719*pi) q[11];
U1q(1.02919406109211*pi,1.9408153321362072*pi) q[12];
U1q(0.369524541285819*pi,1.598538957557048*pi) q[13];
U1q(0.653297903546082*pi,0.210817930334702*pi) q[14];
U1q(3.475876291242092*pi,1.4530199340397218*pi) q[15];
U1q(0.3878152287293*pi,1.9736109081031812*pi) q[16];
U1q(0.111483176330086*pi,1.815167969442451*pi) q[17];
U1q(0.73866997373556*pi,0.20286180941631*pi) q[18];
U1q(0.0702495264166125*pi,1.415181708042679*pi) q[19];
U1q(0.534788067113009*pi,0.659008275739992*pi) q[20];
U1q(1.66912456955687*pi,0.161933439288356*pi) q[21];
U1q(0.797507816141234*pi,1.21380331923359*pi) q[22];
U1q(3.279825186938063*pi,1.484299085924127*pi) q[23];
U1q(0.092897340037183*pi,1.168494548800871*pi) q[24];
U1q(1.11704157256656*pi,1.725162938079308*pi) q[25];
U1q(0.204940762112535*pi,1.2980601720394112*pi) q[26];
U1q(1.85438224990637*pi,0.661135986722029*pi) q[27];
U1q(0.462339122450147*pi,0.557637589458523*pi) q[28];
U1q(0.293247497182333*pi,0.367441003189171*pi) q[29];
U1q(1.42303784138745*pi,0.367902269919382*pi) q[30];
U1q(3.353133564506943*pi,0.85953621245598*pi) q[31];
U1q(1.34815469254214*pi,1.572492884951464*pi) q[32];
U1q(1.44968636493092*pi,0.9930743100016199*pi) q[33];
U1q(3.051155792574915*pi,0.146072364837174*pi) q[34];
U1q(0.642190034735442*pi,0.398195301204913*pi) q[35];
U1q(1.44536310630973*pi,0.98546815844419*pi) q[36];
U1q(3.639078712613505*pi,0.0346824258763572*pi) q[37];
U1q(3.784187844119591*pi,1.841537965315574*pi) q[38];
U1q(1.03208338482179*pi,0.68537005466293*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[37];
RZZ(0.5*pi) q[2],q[34];
RZZ(0.5*pi) q[3],q[33];
RZZ(0.5*pi) q[4],q[25];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[13],q[18];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[16],q[19];
RZZ(0.5*pi) q[29],q[17];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[23],q[31];
RZZ(0.5*pi) q[38],q[26];
RZZ(0.5*pi) q[28],q[39];
U1q(3.066564094201265*pi,1.4401829643611586*pi) q[0];
U1q(1.5219735013556*pi,1.297823539403753*pi) q[1];
U1q(3.867947749712622*pi,0.0699635202310187*pi) q[2];
U1q(3.486238810234766*pi,0.8177488974884163*pi) q[3];
U1q(0.299058428317944*pi,0.25254027623171993*pi) q[4];
U1q(0.220285916129051*pi,0.28825435021074997*pi) q[5];
U1q(1.75909314065779*pi,1.362903130004067*pi) q[6];
U1q(0.0783673411921988*pi,0.253275117200878*pi) q[7];
U1q(0.528344023273895*pi,0.50599668557043*pi) q[8];
U1q(1.33791517455783*pi,0.9440814859097499*pi) q[9];
U1q(1.9281667386822*pi,0.2096872084213116*pi) q[10];
U1q(1.15546944324588*pi,1.0204522900441702*pi) q[11];
U1q(1.52914188917238*pi,1.4528532693802998*pi) q[12];
U1q(0.681556854866279*pi,1.30225408038108*pi) q[13];
U1q(0.169526330167391*pi,0.51453884339502*pi) q[14];
U1q(1.21696060307172*pi,1.9541865375658052*pi) q[15];
U1q(1.0910197070737*pi,1.98710223145897*pi) q[16];
U1q(0.122923331611362*pi,0.27903390349949*pi) q[17];
U1q(0.386285738137542*pi,1.711397031192454*pi) q[18];
U1q(1.36732175707872*pi,0.49732736872686*pi) q[19];
U1q(0.105418416485896*pi,1.428689846784396*pi) q[20];
U1q(3.228421772121921*pi,0.9198267854476125*pi) q[21];
U1q(0.407869293381459*pi,1.876794043490894*pi) q[22];
U1q(1.32822817570152*pi,0.1558176628562553*pi) q[23];
U1q(1.79093950359706*pi,0.8723722863674599*pi) q[24];
U1q(3.322223531743859*pi,0.9823998478705005*pi) q[25];
U1q(0.536863145184213*pi,0.249668358107981*pi) q[26];
U1q(1.71589632093277*pi,1.784106993054*pi) q[27];
U1q(1.67748591885917*pi,0.298059938758324*pi) q[28];
U1q(1.27670414719758*pi,0.726908316395556*pi) q[29];
U1q(1.7128585434071*pi,0.19510721935426345*pi) q[30];
U1q(3.274929362075473*pi,1.1978459232266192*pi) q[31];
U1q(3.812311009288819*pi,1.1836921884269689*pi) q[32];
U1q(3.315764962331917*pi,1.0371060338412619*pi) q[33];
U1q(1.25084288765037*pi,1.0709532592418185*pi) q[34];
U1q(3.476933044956989*pi,0.0227351638814505*pi) q[35];
U1q(3.571489985548581*pi,1.4557157794782203*pi) q[36];
U1q(3.1508769733152358*pi,0.3572685469965453*pi) q[37];
U1q(3.6684458715460098*pi,0.6711605096298126*pi) q[38];
U1q(1.03412553789584*pi,0.41297484582882205*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[20],q[1];
RZZ(0.5*pi) q[2],q[22];
RZZ(0.5*pi) q[3],q[21];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[23],q[10];
RZZ(0.5*pi) q[11],q[18];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[24],q[13];
RZZ(0.5*pi) q[14],q[37];
RZZ(0.5*pi) q[16],q[15];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[25],q[31];
RZZ(0.5*pi) q[39],q[27];
RZZ(0.5*pi) q[29],q[32];
RZZ(0.5*pi) q[34],q[30];
RZZ(0.5*pi) q[35],q[36];
U1q(3.549784065410505*pi,0.6260528836660739*pi) q[0];
U1q(3.588442650006978*pi,1.6815834218196408*pi) q[1];
U1q(1.35733281541655*pi,1.5437871881930199*pi) q[2];
U1q(3.607406841713067*pi,0.4030484052931923*pi) q[3];
U1q(3.027021977131114*pi,1.03619976228389*pi) q[4];
U1q(1.73735419039538*pi,1.32128510958499*pi) q[5];
U1q(3.297548379421622*pi,1.1393773098013396*pi) q[6];
U1q(0.338607560182068*pi,1.668768463745268*pi) q[7];
U1q(3.717695696106128*pi,1.195621634122257*pi) q[8];
U1q(3.519213956943533*pi,0.21914409807952606*pi) q[9];
U1q(1.38580191909687*pi,0.9155862362496976*pi) q[10];
U1q(3.070603881578069*pi,0.33997776081923803*pi) q[11];
U1q(0.418367167361903*pi,0.6883559326426223*pi) q[12];
U1q(1.64591824017464*pi,1.4531736945433096*pi) q[13];
U1q(0.594400312942993*pi,1.6940877975324602*pi) q[14];
U1q(1.25886852622404*pi,1.9684983828677152*pi) q[15];
U1q(3.495311996015392*pi,1.186744409701494*pi) q[16];
U1q(1.59538940981585*pi,0.21920392798991983*pi) q[17];
U1q(0.257621401940699*pi,1.81248130555969*pi) q[18];
U1q(3.594900370805791*pi,0.09561190274955522*pi) q[19];
U1q(1.76200102987892*pi,1.4280687279868198*pi) q[20];
U1q(3.390346062796035*pi,0.8268286075477924*pi) q[21];
U1q(0.385845262332028*pi,1.441425640462462*pi) q[22];
U1q(0.338159995404855*pi,1.8548843187293755*pi) q[23];
U1q(3.208311114107045*pi,1.8829709012066709*pi) q[24];
U1q(3.1243776347407017*pi,1.2506166931547804*pi) q[25];
U1q(0.302963427037614*pi,1.23112174389746*pi) q[26];
U1q(0.468686125080435*pi,1.866053923121811*pi) q[27];
U1q(1.64355992648358*pi,0.28178365136097666*pi) q[28];
U1q(3.861088680714923*pi,1.0120508842383624*pi) q[29];
U1q(1.47144727875327*pi,1.9596850962541315*pi) q[30];
U1q(3.473097735680928*pi,1.2416560825820493*pi) q[31];
U1q(3.818399372627245*pi,1.3874160583586508*pi) q[32];
U1q(1.21193779168575*pi,0.906869458121415*pi) q[33];
U1q(1.21535152565711*pi,1.7419246218818643*pi) q[34];
U1q(1.52599863543391*pi,1.376611923574281*pi) q[35];
U1q(0.293758485163122*pi,1.50002128982739*pi) q[36];
U1q(3.769985818032493*pi,1.8304628941195102*pi) q[37];
U1q(1.55945444330522*pi,1.0893192015187463*pi) q[38];
U1q(1.45724151226566*pi,1.976462166318882*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[4],q[26];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[21],q[6];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[8],q[36];
RZZ(0.5*pi) q[9],q[34];
RZZ(0.5*pi) q[10],q[22];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[13],q[37];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[25],q[17];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[30],q[27];
RZZ(0.5*pi) q[28],q[35];
RZZ(0.5*pi) q[32],q[33];
U1q(0.702769109353906*pi,1.815838780373944*pi) q[0];
U1q(3.851811861232928*pi,1.1450081871811184*pi) q[1];
U1q(1.31923735906568*pi,0.4472854937021489*pi) q[2];
U1q(3.437275667361334*pi,1.884702626126841*pi) q[3];
U1q(3.8439012345854158*pi,1.8260852257867777*pi) q[4];
U1q(1.31833070611618*pi,1.9828104562061144*pi) q[5];
U1q(3.273504776932176*pi,0.03647799731832446*pi) q[6];
U1q(1.62871331128577*pi,0.6650705203534399*pi) q[7];
U1q(3.9735326237417747*pi,1.589715336481187*pi) q[8];
U1q(1.61920597403716*pi,0.9312754280625881*pi) q[9];
U1q(1.81205766876093*pi,1.3743483891500325*pi) q[10];
U1q(1.5734472373339*pi,1.6791715242070584*pi) q[11];
U1q(1.5073522452924*pi,0.9076253350550525*pi) q[12];
U1q(3.82274702106199*pi,1.1547361410177022*pi) q[13];
U1q(1.38211443572049*pi,1.5945392501115099*pi) q[14];
U1q(1.21351368255384*pi,0.41650883437757114*pi) q[15];
U1q(1.0991920555155*pi,1.8551877236898244*pi) q[16];
U1q(1.34304406490771*pi,0.6761415011771872*pi) q[17];
U1q(1.656217826101*pi,1.8847765024509302*pi) q[18];
U1q(1.43833667418711*pi,0.45974423805370446*pi) q[19];
U1q(1.63949321513244*pi,0.6476250816416349*pi) q[20];
U1q(3.17437525704607*pi,1.9467398437450019*pi) q[21];
U1q(1.50116994972116*pi,0.663864246635353*pi) q[22];
U1q(0.694467626923437*pi,1.1006157498500553*pi) q[23];
U1q(0.097933364641084*pi,1.561269015355351*pi) q[24];
U1q(3.544797545496502*pi,1.3303578878461404*pi) q[25];
U1q(1.52788319401891*pi,1.04154856968188*pi) q[26];
U1q(0.66921317258594*pi,1.341869823603791*pi) q[27];
U1q(0.505925250543034*pi,1.3287518138089447*pi) q[28];
U1q(3.238063712753935*pi,0.5316687920334924*pi) q[29];
U1q(3.851820381099338*pi,1.5779871640881544*pi) q[30];
U1q(3.232696894301172*pi,0.3379333813986436*pi) q[31];
U1q(1.40719610582158*pi,1.20608013094215*pi) q[32];
U1q(1.40945176294838*pi,0.31219791689525866*pi) q[33];
U1q(3.742030003436885*pi,1.6178671322400904*pi) q[34];
U1q(1.67281149043597*pi,1.8576484554966506*pi) q[35];
U1q(1.64386323085174*pi,1.7043192441789703*pi) q[36];
U1q(3.672127811545835*pi,0.9217441821192502*pi) q[37];
U1q(0.314695502229745*pi,1.0035141169644812*pi) q[38];
U1q(1.92310239662779*pi,1.0974113850500409*pi) q[39];
RZZ(0.5*pi) q[0],q[36];
RZZ(0.5*pi) q[1],q[26];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[15];
RZZ(0.5*pi) q[4],q[38];
RZZ(0.5*pi) q[9],q[6];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[8],q[21];
RZZ(0.5*pi) q[28],q[10];
RZZ(0.5*pi) q[24],q[11];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[14],q[13];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[33],q[17];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[31],q[35];
RZZ(0.5*pi) q[37],q[39];
U1q(0.449683125097251*pi,0.7112924368587938*pi) q[0];
U1q(1.38236445212899*pi,1.454544053196249*pi) q[1];
U1q(1.39708692033064*pi,1.0208235091493623*pi) q[2];
U1q(1.42261436854895*pi,0.1886081616802271*pi) q[3];
U1q(1.59447916797642*pi,0.4797799307202082*pi) q[4];
U1q(0.687778723229879*pi,0.3775054637607944*pi) q[5];
U1q(3.59812932334777*pi,1.029281915270957*pi) q[6];
U1q(1.52570683365919*pi,0.3481909735763251*pi) q[7];
U1q(1.62451806188652*pi,0.5310776866675249*pi) q[8];
U1q(0.436846030532685*pi,0.16052270754914888*pi) q[9];
U1q(0.759400179649106*pi,0.16269770630395253*pi) q[10];
U1q(3.393675478851649*pi,1.9271153470192273*pi) q[11];
U1q(1.30032838123466*pi,1.023519148100391*pi) q[12];
U1q(1.5563258351536*pi,0.0787491760571295*pi) q[13];
U1q(1.44146935626417*pi,0.708641521387678*pi) q[14];
U1q(0.607065016006823*pi,1.201061301064381*pi) q[15];
U1q(0.637251021099394*pi,1.0062134815047736*pi) q[16];
U1q(0.479617792642381*pi,0.7693783548340072*pi) q[17];
U1q(1.3862440991433*pi,0.743312135036041*pi) q[18];
U1q(0.731898911762354*pi,1.1916934313396643*pi) q[19];
U1q(0.786234134819382*pi,0.24406644708082492*pi) q[20];
U1q(1.06194866353367*pi,1.2183095794053758*pi) q[21];
U1q(1.69948891738437*pi,1.3277287317006294*pi) q[22];
U1q(0.737936847364053*pi,0.7584827644247758*pi) q[23];
U1q(0.65789297860459*pi,0.35315951570201065*pi) q[24];
U1q(1.28299964201865*pi,1.069308940482903*pi) q[25];
U1q(1.37514370529541*pi,0.741122087403344*pi) q[26];
U1q(0.63185373607045*pi,1.4597544756085608*pi) q[27];
U1q(0.484014931897058*pi,0.7879864514956627*pi) q[28];
U1q(3.736353268501762*pi,0.9647108472362746*pi) q[29];
U1q(1.89489475314653*pi,1.460350271060249*pi) q[30];
U1q(0.747383176714568*pi,1.8476680319968737*pi) q[31];
U1q(0.768152621713417*pi,1.5294416103330928*pi) q[32];
U1q(1.78311339523228*pi,0.559376458857276*pi) q[33];
U1q(1.13661572420089*pi,0.8319880719191275*pi) q[34];
U1q(1.63834888302398*pi,0.23080352865825549*pi) q[35];
U1q(1.47340393027828*pi,1.8183538926031373*pi) q[36];
U1q(1.81735151019318*pi,1.9021778276625696*pi) q[37];
U1q(0.576319096791119*pi,1.9413543716531807*pi) q[38];
U1q(0.38337227779845*pi,1.269991802537061*pi) q[39];
rz(3.288707563141206*pi) q[0];
rz(2.545455946803751*pi) q[1];
rz(0.9791764908506377*pi) q[2];
rz(1.811391838319773*pi) q[3];
rz(3.520220069279792*pi) q[4];
rz(1.6224945362392056*pi) q[5];
rz(0.9707180847290432*pi) q[6];
rz(1.6518090264236749*pi) q[7];
rz(1.4689223133324751*pi) q[8];
rz(3.839477292450851*pi) q[9];
rz(1.8373022936960475*pi) q[10];
rz(0.07288465298077274*pi) q[11];
rz(0.9764808518996091*pi) q[12];
rz(3.9212508239428705*pi) q[13];
rz(3.291358478612322*pi) q[14];
rz(0.7989386989356191*pi) q[15];
rz(0.9937865184952264*pi) q[16];
rz(3.230621645165993*pi) q[17];
rz(3.256687864963959*pi) q[18];
rz(0.8083065686603357*pi) q[19];
rz(3.755933552919175*pi) q[20];
rz(2.7816904205946242*pi) q[21];
rz(0.6722712682993707*pi) q[22];
rz(3.241517235575224*pi) q[23];
rz(1.6468404842979893*pi) q[24];
rz(2.930691059517097*pi) q[25];
rz(1.258877912596656*pi) q[26];
rz(2.5402455243914392*pi) q[27];
rz(1.2120135485043373*pi) q[28];
rz(3.0352891527637254*pi) q[29];
rz(2.539649728939751*pi) q[30];
rz(2.1523319680031263*pi) q[31];
rz(2.470558389666907*pi) q[32];
rz(3.440623541142724*pi) q[33];
rz(1.1680119280808725*pi) q[34];
rz(3.7691964713417443*pi) q[35];
rz(0.1816461073968627*pi) q[36];
rz(0.0978221723374304*pi) q[37];
rz(0.0586456283468193*pi) q[38];
rz(2.730008197462939*pi) q[39];
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