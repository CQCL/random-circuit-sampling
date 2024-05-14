OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.780941958697101*pi,0.637013635114078*pi) q[0];
U1q(0.611260347238211*pi,1.800716160523602*pi) q[1];
U1q(0.0374661675107036*pi,0.8020621003028501*pi) q[2];
U1q(0.409142305138551*pi,0.6760343035535299*pi) q[3];
U1q(0.280341876824773*pi,0.691560807780673*pi) q[4];
U1q(0.360643354874956*pi,1.069556633300526*pi) q[5];
U1q(0.121960919291785*pi,1.24519360658405*pi) q[6];
U1q(0.470852037155247*pi,1.802977330612167*pi) q[7];
U1q(0.721350599156994*pi,1.328589457177622*pi) q[8];
U1q(0.421757348528719*pi,0.4995754450449099*pi) q[9];
U1q(0.894691166346193*pi,1.649760825835814*pi) q[10];
U1q(0.579648547124785*pi,0.591063556942384*pi) q[11];
U1q(0.19288353633065*pi,1.381071000791596*pi) q[12];
U1q(0.644715945061678*pi,0.493713987564919*pi) q[13];
U1q(0.481219655860578*pi,0.54179562821021*pi) q[14];
U1q(0.507228683680505*pi,0.581896308991404*pi) q[15];
U1q(0.853953252193561*pi,0.636863403631651*pi) q[16];
U1q(0.650530227747966*pi,1.11986507681128*pi) q[17];
U1q(0.340157566508165*pi,1.674207172187141*pi) q[18];
U1q(0.413760271384143*pi,0.94134691421045*pi) q[19];
U1q(0.300084279316857*pi,0.166662228780133*pi) q[20];
U1q(0.655940909587143*pi,0.408147289546951*pi) q[21];
U1q(0.682043545881755*pi,0.681774498476564*pi) q[22];
U1q(0.201951548842847*pi,1.750803358423047*pi) q[23];
U1q(0.464578393579665*pi,0.218243306643471*pi) q[24];
U1q(0.786387405091996*pi,0.525792673631904*pi) q[25];
U1q(0.651513837938005*pi,1.11450141538414*pi) q[26];
U1q(0.464752932410665*pi,1.624767203755321*pi) q[27];
U1q(0.498380462489052*pi,1.187986337446*pi) q[28];
U1q(0.874231528792707*pi,1.9958149130212144*pi) q[29];
U1q(0.447876086269078*pi,1.4653216087019931*pi) q[30];
U1q(0.460870477561724*pi,1.600920796376107*pi) q[31];
U1q(0.871901181564434*pi,1.606493789337381*pi) q[32];
U1q(0.457402668080311*pi,1.628614774649314*pi) q[33];
U1q(0.436143378908546*pi,0.665138603238609*pi) q[34];
U1q(0.233798367700348*pi,1.933305511366605*pi) q[35];
U1q(0.437325616791681*pi,0.862001218476117*pi) q[36];
U1q(0.73148871555769*pi,1.321154154685108*pi) q[37];
U1q(0.314589741130036*pi,1.393160897025461*pi) q[38];
U1q(0.291404286162369*pi,1.186744084645085*pi) q[39];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[2],q[9];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[32];
RZZ(0.5*pi) q[7],q[37];
RZZ(0.5*pi) q[33],q[8];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[14],q[12];
RZZ(0.5*pi) q[13],q[35];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[23],q[24];
RZZ(0.5*pi) q[38],q[25];
RZZ(0.5*pi) q[31],q[28];
RZZ(0.5*pi) q[34],q[30];
U1q(0.526975583275999*pi,1.671826460298019*pi) q[0];
U1q(0.207372590628203*pi,0.7087627196253701*pi) q[1];
U1q(0.392440776350503*pi,1.1598556636586*pi) q[2];
U1q(0.802609736811493*pi,0.26755665662264994*pi) q[3];
U1q(0.822349370202696*pi,1.0532359248605099*pi) q[4];
U1q(0.128308978570291*pi,0.25014152144636004*pi) q[5];
U1q(0.533947596893801*pi,0.5501162874076302*pi) q[6];
U1q(0.649012873097977*pi,1.9738927495129102*pi) q[7];
U1q(0.629528652017884*pi,0.9934283418933401*pi) q[8];
U1q(0.557029550876378*pi,0.5242546776182699*pi) q[9];
U1q(0.534884735860226*pi,0.8349653254408298*pi) q[10];
U1q(0.796689929425371*pi,1.0016597034711339*pi) q[11];
U1q(0.523692079612764*pi,0.37649516323156007*pi) q[12];
U1q(0.181423949107174*pi,1.883980011733241*pi) q[13];
U1q(0.717897006108553*pi,1.82495947014392*pi) q[14];
U1q(0.737827546681082*pi,1.058885788183318*pi) q[15];
U1q(0.367815209489981*pi,1.98264083736415*pi) q[16];
U1q(0.355186920947465*pi,0.385035121537677*pi) q[17];
U1q(0.296519636620137*pi,0.7253823608430499*pi) q[18];
U1q(0.670121843341049*pi,1.8734604841139504*pi) q[19];
U1q(0.185500377953728*pi,1.01262166765753*pi) q[20];
U1q(0.740086730513403*pi,1.263806874805651*pi) q[21];
U1q(0.900361035787098*pi,1.0106833050217299*pi) q[22];
U1q(0.0780791555803017*pi,1.5185856024646198*pi) q[23];
U1q(0.260689960854746*pi,1.3635167110593498*pi) q[24];
U1q(0.724833364376599*pi,0.12294714746594*pi) q[25];
U1q(0.259168879472435*pi,0.31732109990414004*pi) q[26];
U1q(0.313517235818*pi,1.08562097299551*pi) q[27];
U1q(0.846000804627991*pi,0.5593760837673301*pi) q[28];
U1q(0.173052493188316*pi,1.57765627527212*pi) q[29];
U1q(0.394228830537021*pi,1.60405907924354*pi) q[30];
U1q(0.578786695349701*pi,1.1754441444650698*pi) q[31];
U1q(0.156710681214314*pi,1.0934301752935598*pi) q[32];
U1q(0.0621127683982574*pi,0.15400290031334007*pi) q[33];
U1q(0.693259981551875*pi,1.9816241797251202*pi) q[34];
U1q(0.313793253017394*pi,0.4035320597393399*pi) q[35];
U1q(0.125227987783214*pi,0.5410332387405901*pi) q[36];
U1q(0.671934086978399*pi,1.3193337284721096*pi) q[37];
U1q(0.431264022561474*pi,0.10938074284720001*pi) q[38];
U1q(0.439207537617741*pi,0.3359348314451598*pi) q[39];
RZZ(0.5*pi) q[0],q[35];
RZZ(0.5*pi) q[1],q[23];
RZZ(0.5*pi) q[2],q[28];
RZZ(0.5*pi) q[30],q[3];
RZZ(0.5*pi) q[29],q[4];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[6],q[25];
RZZ(0.5*pi) q[10],q[8];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[31],q[14];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[17],q[18];
RZZ(0.5*pi) q[19],q[32];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[38],q[22];
RZZ(0.5*pi) q[34],q[24];
U1q(0.894343679655739*pi,0.8162370613672201*pi) q[0];
U1q(0.842411239330962*pi,1.7328536247579*pi) q[1];
U1q(0.862878542852665*pi,1.8876118800547799*pi) q[2];
U1q(0.376658353397305*pi,1.13359975950838*pi) q[3];
U1q(0.733439654575118*pi,0.1132161077414402*pi) q[4];
U1q(0.291447271730959*pi,1.4526728798625896*pi) q[5];
U1q(0.471353144777553*pi,1.5755264764769592*pi) q[6];
U1q(0.607389044134243*pi,0.2175562762396499*pi) q[7];
U1q(0.363471958356924*pi,1.2389783547464104*pi) q[8];
U1q(0.801935169261709*pi,0.5554253373401901*pi) q[9];
U1q(0.553833402448228*pi,1.65536890115553*pi) q[10];
U1q(0.531430067376836*pi,0.8373688600770199*pi) q[11];
U1q(0.348116909845104*pi,1.40063818217962*pi) q[12];
U1q(0.279239418036513*pi,1.27106255946296*pi) q[13];
U1q(0.823146748579285*pi,1.69329624204435*pi) q[14];
U1q(0.432343424316373*pi,1.1375111342316702*pi) q[15];
U1q(0.484464909228102*pi,0.30317951635744*pi) q[16];
U1q(0.493488750528462*pi,0.61307686350455*pi) q[17];
U1q(0.840349311091637*pi,1.5001571258941597*pi) q[18];
U1q(0.486494052885611*pi,1.6573877382288096*pi) q[19];
U1q(0.375188447665146*pi,0.39161110266419996*pi) q[20];
U1q(0.107952721661904*pi,0.06138578186628996*pi) q[21];
U1q(0.413798461937077*pi,0.4428025566278402*pi) q[22];
U1q(0.321487482429317*pi,0.18901501788692965*pi) q[23];
U1q(0.804147457015522*pi,0.66216199708738*pi) q[24];
U1q(0.900979962316793*pi,1.6791663156878398*pi) q[25];
U1q(0.2759828052256*pi,1.14615215444227*pi) q[26];
U1q(0.609380179944293*pi,0.5534565699123597*pi) q[27];
U1q(0.522401802600792*pi,0.48409806265686006*pi) q[28];
U1q(0.687806275477839*pi,1.2207979968963096*pi) q[29];
U1q(0.0222220412153672*pi,1.1077792536514197*pi) q[30];
U1q(0.530424500003073*pi,0.47057689794074964*pi) q[31];
U1q(0.12197452358052*pi,0.83463945006927*pi) q[32];
U1q(0.268632999622357*pi,1.9907225464259897*pi) q[33];
U1q(0.5375912434254*pi,0.8862981650794204*pi) q[34];
U1q(0.390461996794532*pi,1.7139668051125003*pi) q[35];
U1q(0.691768652403274*pi,1.1864471213151004*pi) q[36];
U1q(0.0308325912083602*pi,1.4765700262805996*pi) q[37];
U1q(0.581704955202053*pi,1.7991185348305399*pi) q[38];
U1q(0.372871135001421*pi,0.02139481057193038*pi) q[39];
RZZ(0.5*pi) q[0],q[22];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[2],q[36];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[4],q[32];
RZZ(0.5*pi) q[5],q[20];
RZZ(0.5*pi) q[7],q[6];
RZZ(0.5*pi) q[8],q[28];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[23],q[11];
RZZ(0.5*pi) q[24],q[12];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[33],q[15];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[27],q[18];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[38],q[21];
RZZ(0.5*pi) q[30],q[26];
RZZ(0.5*pi) q[39],q[37];
U1q(0.543523419802462*pi,0.10272177172697994*pi) q[0];
U1q(0.262404427076234*pi,1.1229052279555898*pi) q[1];
U1q(0.55827907600602*pi,1.2912665381645203*pi) q[2];
U1q(0.396272330689357*pi,1.4186116868854395*pi) q[3];
U1q(0.526487549389452*pi,1.5522272282970793*pi) q[4];
U1q(0.870612954678947*pi,0.7315196399624799*pi) q[5];
U1q(0.844261161732317*pi,0.7589452529611105*pi) q[6];
U1q(0.554154271604871*pi,1.2281343360005401*pi) q[7];
U1q(0.673466210351765*pi,1.4905515233757498*pi) q[8];
U1q(0.891073034332487*pi,0.6961393668740401*pi) q[9];
U1q(0.858913551955299*pi,0.84411289455005*pi) q[10];
U1q(0.564173063379378*pi,1.9278614692931102*pi) q[11];
U1q(0.621836437037464*pi,1.8937243883300203*pi) q[12];
U1q(0.727640168609228*pi,1.2037780732092598*pi) q[13];
U1q(0.275366732688231*pi,0.3129744989183596*pi) q[14];
U1q(0.174711632970472*pi,1.5588475864252196*pi) q[15];
U1q(0.519387339575532*pi,1.0703510730798698*pi) q[16];
U1q(0.380656391667897*pi,1.2788169849051503*pi) q[17];
U1q(0.588194136845709*pi,1.9466798285558404*pi) q[18];
U1q(0.703237616818956*pi,0.5574751043282191*pi) q[19];
U1q(0.423233314292913*pi,1.20147994403127*pi) q[20];
U1q(0.645126945323074*pi,1.20810825605587*pi) q[21];
U1q(0.923703576640642*pi,1.7399298536818302*pi) q[22];
U1q(0.253753383115711*pi,1.3976520689646197*pi) q[23];
U1q(0.880920448653453*pi,1.4626053914841393*pi) q[24];
U1q(0.499045130264219*pi,0.7314950121766497*pi) q[25];
U1q(0.497844398979543*pi,0.2654649121270998*pi) q[26];
U1q(0.137792395585534*pi,1.1409411553174191*pi) q[27];
U1q(0.852301687084036*pi,1.6277178212090604*pi) q[28];
U1q(0.294774422486434*pi,0.6331884433850297*pi) q[29];
U1q(0.429654059889178*pi,0.7712066570193503*pi) q[30];
U1q(0.394292737924875*pi,0.2871585329324091*pi) q[31];
U1q(0.522034660450058*pi,1.3636228964975494*pi) q[32];
U1q(0.540075799013428*pi,1.7967528268184605*pi) q[33];
U1q(0.344694568200349*pi,0.57290971796328*pi) q[34];
U1q(0.891273687323841*pi,0.8091573970678603*pi) q[35];
U1q(0.661943502484859*pi,1.08545857824911*pi) q[36];
U1q(0.688433002720186*pi,1.4470462859042001*pi) q[37];
U1q(0.821298114061048*pi,0.3365075249499201*pi) q[38];
U1q(0.621848571145328*pi,0.3791882084135496*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[38],q[3];
RZZ(0.5*pi) q[11],q[4];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[34],q[7];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[10],q[36];
RZZ(0.5*pi) q[30],q[14];
RZZ(0.5*pi) q[15],q[26];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[17],q[32];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[19],q[35];
RZZ(0.5*pi) q[27],q[20];
RZZ(0.5*pi) q[31],q[24];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[37],q[28];
U1q(0.208077787895909*pi,1.8971674858217504*pi) q[0];
U1q(0.568752253844922*pi,0.5224386478444796*pi) q[1];
U1q(0.500494680959343*pi,0.29637952036403004*pi) q[2];
U1q(0.556322071414544*pi,1.8344047101482008*pi) q[3];
U1q(0.531081929214365*pi,1.7916273982819995*pi) q[4];
U1q(0.44683823647243*pi,0.7541200319952193*pi) q[5];
U1q(0.288359267628662*pi,0.8161190853971991*pi) q[6];
U1q(0.354312115380051*pi,1.6961878382348798*pi) q[7];
U1q(0.715788875659339*pi,0.42669289323490034*pi) q[8];
U1q(0.337203930670074*pi,0.6573815930849003*pi) q[9];
U1q(0.588661711953302*pi,1.99647015387551*pi) q[10];
U1q(0.509276594577948*pi,1.9321807349503093*pi) q[11];
U1q(0.81530326540951*pi,0.5718849265365495*pi) q[12];
U1q(0.20873866505975*pi,0.041993670946210315*pi) q[13];
U1q(0.425922263807787*pi,0.3943450331392597*pi) q[14];
U1q(0.258725362632561*pi,0.8916904031310793*pi) q[15];
U1q(0.86988797390389*pi,0.44239459078119037*pi) q[16];
U1q(0.325571856238456*pi,0.6513523832699502*pi) q[17];
U1q(0.549055132386404*pi,0.9917415622581203*pi) q[18];
U1q(0.575548928204534*pi,0.5541676370554001*pi) q[19];
U1q(0.1600409355394*pi,1.0256941913785003*pi) q[20];
U1q(0.77474147922134*pi,1.8908684852031703*pi) q[21];
U1q(0.985120696677083*pi,1.8637006281730804*pi) q[22];
U1q(0.690721439815852*pi,1.0847939705282101*pi) q[23];
U1q(0.289676324621039*pi,0.7302195170915002*pi) q[24];
U1q(0.323706088057931*pi,1.0749262135186797*pi) q[25];
U1q(0.473637869439778*pi,1.6213946567050606*pi) q[26];
U1q(0.845672572394388*pi,0.902443191483*pi) q[27];
U1q(0.331491569564801*pi,0.7941001522051199*pi) q[28];
U1q(0.625147630846349*pi,1.0190965582328104*pi) q[29];
U1q(0.60646618570861*pi,0.18218136907012017*pi) q[30];
U1q(0.733207552593285*pi,1.3056829684192*pi) q[31];
U1q(0.670048128531018*pi,0.5099605094644009*pi) q[32];
U1q(0.235064931802269*pi,1.4775998798027992*pi) q[33];
U1q(0.391715195820506*pi,0.9791198564801995*pi) q[34];
U1q(0.692548588380181*pi,0.07594345105870026*pi) q[35];
U1q(0.303977707905112*pi,1.2173172175206997*pi) q[36];
U1q(0.384265910033806*pi,1.6064532678407009*pi) q[37];
U1q(0.174001979199467*pi,0.5418658978559705*pi) q[38];
U1q(0.348382718183655*pi,0.18100586769113036*pi) q[39];
RZZ(0.5*pi) q[19],q[0];
RZZ(0.5*pi) q[29],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[24],q[3];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[13],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[9],q[18];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[36],q[21];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[33],q[25];
RZZ(0.5*pi) q[27],q[32];
RZZ(0.5*pi) q[38],q[37];
U1q(0.183338381676544*pi,0.9653701647148996*pi) q[0];
U1q(0.726054539383914*pi,0.7542756603979406*pi) q[1];
U1q(0.11309054044289*pi,0.34299750065409995*pi) q[2];
U1q(0.183530774963143*pi,0.33452744960199965*pi) q[3];
U1q(0.266823958723404*pi,0.6119439957551993*pi) q[4];
U1q(0.857110187042286*pi,0.6094779441607994*pi) q[5];
U1q(0.323862452898427*pi,0.17381943118069998*pi) q[6];
U1q(0.431740818426159*pi,1.3701084453790102*pi) q[7];
U1q(0.243589635543805*pi,1.5711752190296995*pi) q[8];
U1q(0.755934920839228*pi,1.4940524116215208*pi) q[9];
U1q(0.170069331533495*pi,0.23523829022686016*pi) q[10];
U1q(0.279474119218478*pi,1.2616694409091007*pi) q[11];
U1q(0.785068741487332*pi,1.6810502896731006*pi) q[12];
U1q(0.456958127009626*pi,1.0603596525188994*pi) q[13];
U1q(0.51461274735149*pi,0.8459178218095005*pi) q[14];
U1q(0.560468096509471*pi,0.19184585079812955*pi) q[15];
U1q(0.29169842502321*pi,0.1980117572840001*pi) q[16];
U1q(0.456734947772277*pi,0.18180050919779944*pi) q[17];
U1q(0.269387177776026*pi,1.6099024626001093*pi) q[18];
U1q(0.511186943807974*pi,1.6316346001100008*pi) q[19];
U1q(0.228775830790082*pi,0.44880510539359975*pi) q[20];
U1q(0.595861241054734*pi,1.6845764781222403*pi) q[21];
U1q(0.557070686468675*pi,1.2059244938336997*pi) q[22];
U1q(0.229845335187639*pi,0.6054326377068993*pi) q[23];
U1q(0.388476483895501*pi,1.3503722931338*pi) q[24];
U1q(0.212441979562362*pi,0.05143088472721047*pi) q[25];
U1q(0.211946176904703*pi,1.7117338105596005*pi) q[26];
U1q(0.643573534551011*pi,1.8193418067153004*pi) q[27];
U1q(0.715597524335808*pi,0.5446743586009006*pi) q[28];
U1q(0.531557162305936*pi,0.32205050577930017*pi) q[29];
U1q(0.625243154495337*pi,1.5846121984731596*pi) q[30];
U1q(0.819960450968541*pi,0.0779565994155007*pi) q[31];
U1q(0.501030895182663*pi,0.7757943358839992*pi) q[32];
U1q(0.0698818290927609*pi,1.2694248893718996*pi) q[33];
U1q(0.519054765043129*pi,0.4906776737200005*pi) q[34];
U1q(0.82631683675668*pi,0.6406144694527995*pi) q[35];
U1q(0.469140343691821*pi,1.0841009911172996*pi) q[36];
U1q(0.835741117937438*pi,0.7219245064769009*pi) q[37];
U1q(0.251399491765249*pi,0.7588378143303007*pi) q[38];
U1q(0.610191218872499*pi,1.8842019317284997*pi) q[39];
RZZ(0.5*pi) q[0],q[12];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[8],q[4];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[6],q[21];
RZZ(0.5*pi) q[29],q[9];
RZZ(0.5*pi) q[10],q[35];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[13],q[32];
RZZ(0.5*pi) q[16],q[14];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[38],q[18];
RZZ(0.5*pi) q[33],q[19];
RZZ(0.5*pi) q[31],q[22];
RZZ(0.5*pi) q[34],q[23];
RZZ(0.5*pi) q[26],q[36];
RZZ(0.5*pi) q[30],q[37];
U1q(0.927407323562545*pi,0.9460210457682994*pi) q[0];
U1q(0.909841044585951*pi,0.7325248291053992*pi) q[1];
U1q(0.434888731531088*pi,0.7029172969478008*pi) q[2];
U1q(0.415165354996549*pi,0.23090878143069915*pi) q[3];
U1q(0.324026286952607*pi,1.2338059251275997*pi) q[4];
U1q(0.813696341300578*pi,0.6795704261927007*pi) q[5];
U1q(0.282959776242709*pi,1.6977433977869012*pi) q[6];
U1q(0.687240359185931*pi,0.8550447398491006*pi) q[7];
U1q(0.857357180994928*pi,1.2593202949668996*pi) q[8];
U1q(0.401387186411435*pi,1.4798036237905006*pi) q[9];
U1q(0.132294135280132*pi,1.3119288901452997*pi) q[10];
U1q(0.816608037363249*pi,0.22337797905539958*pi) q[11];
U1q(0.205317800345703*pi,0.15618405877659924*pi) q[12];
U1q(0.37627118359078*pi,1.2189964037793999*pi) q[13];
U1q(0.67794253509192*pi,1.3325533620685999*pi) q[14];
U1q(0.910975700876956*pi,1.0437446398551007*pi) q[15];
U1q(0.370947393888387*pi,1.1673639558653992*pi) q[16];
U1q(0.39056653054799*pi,1.2667673141709006*pi) q[17];
U1q(0.8211076094337*pi,0.6112203751554208*pi) q[18];
U1q(0.929642176947466*pi,1.195689908917501*pi) q[19];
U1q(0.105707517744866*pi,0.6670180044651008*pi) q[20];
U1q(0.232865098727384*pi,0.04130064868160055*pi) q[21];
U1q(0.609405786951898*pi,0.5580096168076603*pi) q[22];
U1q(0.621365504464427*pi,0.6931523579517993*pi) q[23];
U1q(0.401875295556997*pi,1.4548022066183997*pi) q[24];
U1q(0.355253193342099*pi,0.6054276526506008*pi) q[25];
U1q(0.0297482287842603*pi,0.5358900368917006*pi) q[26];
U1q(0.822838912461054*pi,1.0662035210849012*pi) q[27];
U1q(0.270111193471885*pi,0.052643776519399665*pi) q[28];
U1q(0.834087702149997*pi,1.2847619447242007*pi) q[29];
U1q(0.802651917166449*pi,0.21932567628673993*pi) q[30];
U1q(0.390326983968831*pi,1.3430140775622998*pi) q[31];
U1q(0.593608797534673*pi,0.4508705759338003*pi) q[32];
U1q(0.319667533486266*pi,1.9819092525096984*pi) q[33];
U1q(0.411343175119232*pi,0.8069323668006003*pi) q[34];
U1q(0.506471100971242*pi,0.8481958189357997*pi) q[35];
U1q(0.739878817404327*pi,0.22541338740830064*pi) q[36];
U1q(0.625991558603198*pi,1.5710448318319*pi) q[37];
U1q(0.529050578515372*pi,1.8056013422772992*pi) q[38];
U1q(0.449583551756163*pi,1.5529650725170008*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[3];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[9],q[5];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[8],q[35];
RZZ(0.5*pi) q[19],q[12];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[34],q[16];
RZZ(0.5*pi) q[17],q[26];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[24],q[20];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[22],q[32];
RZZ(0.5*pi) q[29],q[23];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[38],q[28];
U1q(0.697033131496998*pi,0.49106517519090076*pi) q[0];
U1q(0.340794569113903*pi,1.0744802473089994*pi) q[1];
U1q(0.44943456850377*pi,1.6795281690953008*pi) q[2];
U1q(0.550501984200149*pi,1.0648605533287991*pi) q[3];
U1q(0.341880840883089*pi,0.12105203035050138*pi) q[4];
U1q(0.706595337456402*pi,0.07645560800309958*pi) q[5];
U1q(0.325085479886412*pi,0.6653464523250001*pi) q[6];
U1q(0.87396331524085*pi,1.1739671571141006*pi) q[7];
U1q(0.521056614417149*pi,1.1327500248793996*pi) q[8];
U1q(0.668772997349577*pi,0.2350825237785994*pi) q[9];
U1q(0.615401396756382*pi,1.1976086681368*pi) q[10];
U1q(0.288215265563276*pi,1.8132404938368012*pi) q[11];
U1q(0.715034752599078*pi,1.4800747393625002*pi) q[12];
U1q(0.673401815076395*pi,0.40134432040430035*pi) q[13];
U1q(0.33229446715988*pi,0.9486596724362002*pi) q[14];
U1q(0.332489378964469*pi,1.7422878762932008*pi) q[15];
U1q(0.775120566151706*pi,1.3066220115702016*pi) q[16];
U1q(0.325565785684787*pi,1.9182475523752984*pi) q[17];
U1q(0.770614750909772*pi,1.5498581750509004*pi) q[18];
U1q(0.171486717040268*pi,1.7515127655407987*pi) q[19];
U1q(0.787325343071841*pi,1.8176880433819989*pi) q[20];
U1q(0.411507428219284*pi,0.2550529341061001*pi) q[21];
U1q(0.438887242302803*pi,0.97360125711735*pi) q[22];
U1q(0.197142154677003*pi,1.1267021822960999*pi) q[23];
U1q(0.353173970162334*pi,0.7508913408403011*pi) q[24];
U1q(0.425278387708156*pi,1.6515817093889993*pi) q[25];
U1q(0.465292064222041*pi,1.5269380364849994*pi) q[26];
U1q(0.446734236416041*pi,0.45363431401059984*pi) q[27];
U1q(0.315506182394304*pi,0.10435653816970003*pi) q[28];
U1q(0.745859933092837*pi,1.8492527897406994*pi) q[29];
U1q(0.364574331437068*pi,1.3013279708897993*pi) q[30];
U1q(0.614399287139114*pi,1.9397880271422991*pi) q[31];
U1q(0.406729424539214*pi,1.3853835389500002*pi) q[32];
U1q(0.0632882932084052*pi,0.8709992137783011*pi) q[33];
U1q(0.2260332001249*pi,1.822395050229499*pi) q[34];
U1q(0.552039340379202*pi,0.9979576539621*pi) q[35];
U1q(0.582507760710892*pi,1.4989468104318995*pi) q[36];
U1q(0.888988485372352*pi,1.1812128274446998*pi) q[37];
U1q(0.66878911535579*pi,0.8823133941488983*pi) q[38];
U1q(0.598348712539943*pi,0.13397352619050018*pi) q[39];
RZZ(0.5*pi) q[0],q[20];
RZZ(0.5*pi) q[1],q[14];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[17],q[4];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[29],q[6];
RZZ(0.5*pi) q[33],q[7];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[11],q[25];
RZZ(0.5*pi) q[12],q[28];
RZZ(0.5*pi) q[13],q[37];
RZZ(0.5*pi) q[15],q[38];
RZZ(0.5*pi) q[16],q[19];
RZZ(0.5*pi) q[32],q[18];
RZZ(0.5*pi) q[22],q[21];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[34],q[26];
RZZ(0.5*pi) q[39],q[27];
RZZ(0.5*pi) q[31],q[35];
U1q(0.396588635343992*pi,1.7046785342862982*pi) q[0];
U1q(0.451968475769339*pi,0.06380794721260052*pi) q[1];
U1q(0.599800136999974*pi,0.6568501441229007*pi) q[2];
U1q(0.802361679805919*pi,1.0001680034703995*pi) q[3];
U1q(0.776284064869211*pi,0.8252069793680015*pi) q[4];
U1q(0.838174141642012*pi,1.0381382580820002*pi) q[5];
U1q(0.450224772626394*pi,1.7492774162200995*pi) q[6];
U1q(0.375771858044676*pi,0.6637583135122007*pi) q[7];
U1q(0.582692687077583*pi,1.447809666045199*pi) q[8];
U1q(0.0764353329018917*pi,0.31850565322499946*pi) q[9];
U1q(0.90423193868208*pi,1.9444224440296*pi) q[10];
U1q(0.734365536519935*pi,0.8455062208033013*pi) q[11];
U1q(0.476942904179869*pi,1.2080598627484989*pi) q[12];
U1q(0.271481866387293*pi,1.4105665614962994*pi) q[13];
U1q(0.814087151685448*pi,1.6089608045795991*pi) q[14];
U1q(0.281483601875162*pi,1.0121404478027998*pi) q[15];
U1q(0.29361134054734*pi,1.2268316049580008*pi) q[16];
U1q(0.33998705847148*pi,1.4761695411304991*pi) q[17];
U1q(0.513480594474988*pi,1.0410969842432003*pi) q[18];
U1q(0.106850773272553*pi,1.687555528038601*pi) q[19];
U1q(0.941680940539701*pi,1.9989285313323997*pi) q[20];
U1q(0.567167301892043*pi,1.9070105470494987*pi) q[21];
U1q(0.23343403141609*pi,1.4463908811917001*pi) q[22];
U1q(0.916507393748953*pi,1.340478444112101*pi) q[23];
U1q(0.47265664883261*pi,0.15920063276969998*pi) q[24];
U1q(0.334822629236288*pi,1.296754923481199*pi) q[25];
U1q(0.220991581380723*pi,0.12493856955709859*pi) q[26];
U1q(0.226131999636305*pi,1.2412980201285997*pi) q[27];
U1q(0.190567873630257*pi,1.6469579017194995*pi) q[28];
U1q(0.714526306661703*pi,1.1314628378336007*pi) q[29];
U1q(0.135863680941197*pi,1.6942055835064984*pi) q[30];
U1q(0.635415430446509*pi,0.4994892360680012*pi) q[31];
U1q(0.633707994326521*pi,1.7264266710505005*pi) q[32];
U1q(0.124390808346672*pi,1.6354947557531005*pi) q[33];
U1q(0.199354227934564*pi,0.348634003878999*pi) q[34];
U1q(0.568341582110669*pi,1.8866253280616014*pi) q[35];
U1q(0.280225310668937*pi,1.9272372352483984*pi) q[36];
U1q(0.472345441093453*pi,0.3992159552717993*pi) q[37];
U1q(0.873786465456189*pi,1.376387644413299*pi) q[38];
U1q(0.310592212833345*pi,0.3437779112943993*pi) q[39];
RZZ(0.5*pi) q[15],q[0];
RZZ(0.5*pi) q[1],q[32];
RZZ(0.5*pi) q[30],q[2];
RZZ(0.5*pi) q[6],q[3];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[13],q[7];
RZZ(0.5*pi) q[8],q[12];
RZZ(0.5*pi) q[9],q[25];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[34],q[18];
RZZ(0.5*pi) q[36],q[20];
RZZ(0.5*pi) q[31],q[21];
RZZ(0.5*pi) q[22],q[37];
RZZ(0.5*pi) q[39],q[23];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[33],q[28];
U1q(0.322117865619956*pi,0.6676055340330009*pi) q[0];
U1q(0.669209501465616*pi,1.8349349769280998*pi) q[1];
U1q(0.26733216013182*pi,0.6177251713796004*pi) q[2];
U1q(0.558320566889026*pi,1.4201451234992*pi) q[3];
U1q(0.469218638445633*pi,0.4106883828874004*pi) q[4];
U1q(0.688882483685567*pi,1.830339723427901*pi) q[5];
U1q(0.904805104879678*pi,0.4987266682786*pi) q[6];
U1q(0.932086129705901*pi,0.10726046614139939*pi) q[7];
U1q(0.349337207033678*pi,0.7466053227968992*pi) q[8];
U1q(0.518965094450555*pi,0.9580698691575016*pi) q[9];
U1q(0.813929519883851*pi,1.8401782443238002*pi) q[10];
U1q(0.840449420423148*pi,0.11860172212810127*pi) q[11];
U1q(0.289301165468687*pi,0.2504638119223017*pi) q[12];
U1q(0.352325453265657*pi,0.9652191688964002*pi) q[13];
U1q(0.593759517024903*pi,0.9130063857985*pi) q[14];
U1q(0.289888145502075*pi,1.4053290185952996*pi) q[15];
U1q(0.815124470768539*pi,0.7949571750929998*pi) q[16];
U1q(0.356525161356925*pi,1.317494768178701*pi) q[17];
U1q(0.510094296929986*pi,0.6910331428526*pi) q[18];
U1q(0.852399673277691*pi,0.32359734487279823*pi) q[19];
U1q(0.313935251053731*pi,0.8710741849260017*pi) q[20];
U1q(0.175669375186845*pi,0.33730105315479975*pi) q[21];
U1q(0.207090181175004*pi,0.5620700620668*pi) q[22];
U1q(0.74351607752266*pi,1.7784956204115012*pi) q[23];
U1q(0.974965462369779*pi,0.5775104377115987*pi) q[24];
U1q(0.752933757312023*pi,0.8377983634820012*pi) q[25];
U1q(0.347244237240733*pi,1.6802925528897*pi) q[26];
U1q(0.382366020188495*pi,0.17026747560539945*pi) q[27];
U1q(0.0959158764826127*pi,0.5937603343706002*pi) q[28];
U1q(0.565437247355106*pi,0.3563765022878993*pi) q[29];
U1q(0.310534844769598*pi,1.6906257659220998*pi) q[30];
U1q(0.709993358629544*pi,0.6821889509314012*pi) q[31];
U1q(0.371365361220372*pi,1.3447682423569987*pi) q[32];
U1q(0.409031387301013*pi,1.6798797563106014*pi) q[33];
U1q(0.122816019007145*pi,0.9223342313039993*pi) q[34];
U1q(0.530204415046381*pi,0.2745491438891001*pi) q[35];
U1q(0.0899607976978561*pi,1.5798275501446*pi) q[36];
U1q(0.119344980858806*pi,1.961962633518901*pi) q[37];
U1q(0.46186715669931*pi,0.02103513630940057*pi) q[38];
U1q(0.735337793517392*pi,1.5710812500935987*pi) q[39];
RZZ(0.5*pi) q[0],q[4];
RZZ(0.5*pi) q[1],q[27];
RZZ(0.5*pi) q[2],q[8];
RZZ(0.5*pi) q[39],q[3];
RZZ(0.5*pi) q[5],q[36];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[29];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[17],q[24];
RZZ(0.5*pi) q[33],q[18];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[26],q[20];
RZZ(0.5*pi) q[21],q[25];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[31],q[38];
U1q(0.58008421144434*pi,0.2604926554210998*pi) q[0];
U1q(0.665030023310999*pi,1.4852203549476997*pi) q[1];
U1q(0.660539003568305*pi,0.5299582248572996*pi) q[2];
U1q(0.802380683933281*pi,0.4567143867786996*pi) q[3];
U1q(0.705504336030582*pi,1.476926962445301*pi) q[4];
U1q(0.526847566924141*pi,0.5713658608034002*pi) q[5];
U1q(0.585921395867572*pi,0.43106785780999957*pi) q[6];
U1q(0.544078979945542*pi,1.1693508666050008*pi) q[7];
U1q(0.137901024601326*pi,0.7018878018988985*pi) q[8];
U1q(0.695662321952545*pi,0.2660498976410004*pi) q[9];
U1q(0.226386967888209*pi,1.9456110328962986*pi) q[10];
U1q(0.23423070216644*pi,1.4698485079264003*pi) q[11];
U1q(0.734306485087614*pi,1.5983563310898994*pi) q[12];
U1q(0.722376394780592*pi,0.16066434657760098*pi) q[13];
U1q(0.783577848982529*pi,0.7049931723108998*pi) q[14];
U1q(0.845751513940855*pi,0.6461363928929984*pi) q[15];
U1q(0.540322999992439*pi,0.6963430564584989*pi) q[16];
U1q(0.551785335836309*pi,0.7669583676592993*pi) q[17];
U1q(0.670662802787001*pi,0.5000650342979007*pi) q[18];
U1q(0.465541017300161*pi,1.7402519909495986*pi) q[19];
U1q(0.415631804856637*pi,0.10511713248000021*pi) q[20];
U1q(0.761772289017051*pi,1.9494933823636984*pi) q[21];
U1q(0.752165222607939*pi,1.9862056509202013*pi) q[22];
U1q(0.40009593514404*pi,1.7560567317593012*pi) q[23];
U1q(0.332383027017382*pi,1.610466652350599*pi) q[24];
U1q(0.0622956688290631*pi,0.1441826883000985*pi) q[25];
U1q(0.297249362677463*pi,0.041970938525100365*pi) q[26];
U1q(0.744971817782709*pi,1.4703802733710987*pi) q[27];
U1q(0.662874390601366*pi,1.9790685770270002*pi) q[28];
U1q(0.334264892708089*pi,1.9409528831507998*pi) q[29];
U1q(0.810125827549479*pi,0.628773668294599*pi) q[30];
U1q(0.686254126592149*pi,1.6898393344632012*pi) q[31];
U1q(0.516067082989822*pi,1.5511117885412986*pi) q[32];
U1q(0.425415160366383*pi,0.07607416385269872*pi) q[33];
U1q(0.348504744527189*pi,1.4441863421824017*pi) q[34];
U1q(0.138660277099502*pi,0.8970868955398004*pi) q[35];
U1q(0.41714221487548*pi,1.6896777529544984*pi) q[36];
U1q(0.460784662349878*pi,0.14630619781990006*pi) q[37];
U1q(0.194382241075566*pi,1.8188307143593008*pi) q[38];
U1q(0.526403455046625*pi,1.9887009034647996*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[3],q[21];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[16],q[5];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[18],q[25];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[22],q[27];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[29],q[38];
RZZ(0.5*pi) q[36],q[35];
U1q(0.241201643576395*pi,1.3270610660617983*pi) q[0];
U1q(0.242641016175344*pi,1.7954308085385016*pi) q[1];
U1q(0.853134689169456*pi,1.4720369988346*pi) q[2];
U1q(0.563604655258048*pi,0.9222783809020996*pi) q[3];
U1q(0.322813988802782*pi,0.6884403832327983*pi) q[4];
U1q(0.700136309603914*pi,0.005657874434898957*pi) q[5];
U1q(0.433397129452208*pi,0.34861701357850094*pi) q[6];
U1q(0.641164576233044*pi,0.6593870748871993*pi) q[7];
U1q(0.800610670839827*pi,1.6684237099571995*pi) q[8];
U1q(0.349213261413484*pi,0.2877736118623986*pi) q[9];
U1q(0.542345086480072*pi,1.9564952446765993*pi) q[10];
U1q(0.439370505034995*pi,1.9643759609899014*pi) q[11];
U1q(0.689018161662714*pi,1.3760075397125*pi) q[12];
U1q(0.400380864822946*pi,1.0302560839981005*pi) q[13];
U1q(0.715731190326801*pi,0.2524384524224992*pi) q[14];
U1q(0.734346967488362*pi,0.8208205819831988*pi) q[15];
U1q(0.566838242136444*pi,0.06830613035939948*pi) q[16];
U1q(0.184988956449503*pi,0.6317307102256997*pi) q[17];
U1q(0.423661688619855*pi,0.5073993159621999*pi) q[18];
U1q(0.453508993506946*pi,1.8697920562762995*pi) q[19];
U1q(0.841099404852939*pi,1.9063362721996988*pi) q[20];
U1q(0.517495812657598*pi,0.025460134760400877*pi) q[21];
U1q(0.538498358046393*pi,0.5856005052261999*pi) q[22];
U1q(0.237576444718964*pi,0.020765777874199642*pi) q[23];
U1q(0.227482582822997*pi,0.6044392208526013*pi) q[24];
U1q(0.433261630574346*pi,1.2964342615065014*pi) q[25];
U1q(0.872753978358254*pi,0.30794654417200107*pi) q[26];
U1q(0.536844826292146*pi,0.9000734561384007*pi) q[27];
U1q(0.734048159830275*pi,0.41505319602169877*pi) q[28];
U1q(0.290616519769793*pi,1.437073801494801*pi) q[29];
U1q(0.801688331605751*pi,1.8512102471147998*pi) q[30];
U1q(0.567785812691887*pi,0.6514354683637009*pi) q[31];
U1q(0.268366605924763*pi,0.7444349931375989*pi) q[32];
U1q(0.930850350752939*pi,0.06546604391889943*pi) q[33];
U1q(0.0784877871382976*pi,1.6411413470429999*pi) q[34];
U1q(0.291563600883933*pi,1.0443369095624*pi) q[35];
U1q(0.40310902575906*pi,0.9938372816300998*pi) q[36];
U1q(0.72003284405434*pi,1.2717440052550018*pi) q[37];
U1q(0.739787033199021*pi,0.7608257424797991*pi) q[38];
U1q(0.710336238883447*pi,0.40576018398180125*pi) q[39];
RZZ(0.5*pi) q[0],q[28];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[4],q[27];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[31],q[6];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[8],q[25];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[39];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[13],q[21];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[33],q[16];
RZZ(0.5*pi) q[17],q[22];
RZZ(0.5*pi) q[23],q[37];
RZZ(0.5*pi) q[38],q[24];
RZZ(0.5*pi) q[29],q[26];
RZZ(0.5*pi) q[30],q[36];
U1q(0.495698117679605*pi,0.9487252200052012*pi) q[0];
U1q(0.981356633509388*pi,0.8774774498204003*pi) q[1];
U1q(0.448503507732951*pi,0.7091897704722996*pi) q[2];
U1q(0.056934614004599*pi,0.6887348479239996*pi) q[3];
U1q(0.360825059746977*pi,1.5269051720933007*pi) q[4];
U1q(0.660690668943777*pi,1.3145277845924*pi) q[5];
U1q(0.36756439144884*pi,0.6532058361437016*pi) q[6];
U1q(0.143724739126668*pi,1.0179164948897998*pi) q[7];
U1q(0.340692137417656*pi,1.9540405285826985*pi) q[8];
U1q(0.806541846036735*pi,0.5057064513655014*pi) q[9];
U1q(0.672148403914415*pi,0.6159052367303985*pi) q[10];
U1q(0.484767909356656*pi,1.9858064760874008*pi) q[11];
U1q(0.549631537243833*pi,1.3921865500383*pi) q[12];
U1q(0.509308289855598*pi,1.3169224142462994*pi) q[13];
U1q(0.26498043308321*pi,0.9213775552507997*pi) q[14];
U1q(0.390196427441172*pi,1.8970490720687998*pi) q[15];
U1q(0.646528773408418*pi,0.5193192295711988*pi) q[16];
U1q(0.611057314374258*pi,0.05952170452389893*pi) q[17];
U1q(0.58045450593701*pi,1.2179855949569003*pi) q[18];
U1q(0.663041296413748*pi,1.6564120698037996*pi) q[19];
U1q(0.437688325749505*pi,1.3351180932899993*pi) q[20];
U1q(0.166078667438964*pi,1.7940899974477986*pi) q[21];
U1q(0.152297326761199*pi,1.4417173969308017*pi) q[22];
U1q(0.369140663973411*pi,0.37692755321440075*pi) q[23];
U1q(0.692305640631095*pi,0.7308451523526998*pi) q[24];
U1q(0.561947946893154*pi,0.15367763490559838*pi) q[25];
U1q(0.644294421190352*pi,1.7646848352712006*pi) q[26];
U1q(0.518319356895955*pi,1.2750522102170976*pi) q[27];
U1q(0.680908677733648*pi,1.7567545340027984*pi) q[28];
U1q(0.817724354321425*pi,0.7400415408832011*pi) q[29];
U1q(0.481838600937576*pi,0.7305777329601*pi) q[30];
U1q(0.667708930490413*pi,1.9948669041411016*pi) q[31];
U1q(0.295924471790972*pi,0.8969128584107011*pi) q[32];
U1q(0.551742790032908*pi,0.9966324861633993*pi) q[33];
U1q(0.134271968598176*pi,0.9167443871576992*pi) q[34];
U1q(0.365923493095242*pi,1.1989700919503008*pi) q[35];
U1q(0.301162420745071*pi,0.7089917511648984*pi) q[36];
U1q(0.660843499771925*pi,0.3796503023853006*pi) q[37];
U1q(0.595482116365963*pi,0.4983726588424986*pi) q[38];
U1q(0.557939859845057*pi,1.966769205335499*pi) q[39];
rz(3.5664515947957014*pi) q[0];
rz(1.1851291231788998*pi) q[1];
rz(3.243672208056001*pi) q[2];
rz(3.6208605822105007*pi) q[3];
rz(0.8759743647470017*pi) q[4];
rz(2.1980253312719995*pi) q[5];
rz(2.4374321022261007*pi) q[6];
rz(2.3002144475983*pi) q[7];
rz(1.7522624957521984*pi) q[8];
rz(1.3773957473718994*pi) q[9];
rz(3.9872974389061007*pi) q[10];
rz(0.7945145021804016*pi) q[11];
rz(3.0797603332284*pi) q[12];
rz(2.5322780931423985*pi) q[13];
rz(1.8706572598714999*pi) q[14];
rz(1.2332479411365007*pi) q[15];
rz(0.1291393906693017*pi) q[16];
rz(0.1751624257527986*pi) q[17];
rz(2.857234123445199*pi) q[18];
rz(1.8631169728679993*pi) q[19];
rz(1.527386959278303*pi) q[20];
rz(1.6951962616406*pi) q[21];
rz(0.006276163296000448*pi) q[22];
rz(3.7168641659872996*pi) q[23];
rz(3.2972030329088007*pi) q[24];
rz(0.6210039255141986*pi) q[25];
rz(3.249441667069899*pi) q[26];
rz(2.0509672648174018*pi) q[27];
rz(1.2570957941677996*pi) q[28];
rz(0.48966224497889854*pi) q[29];
rz(0.16434600543599842*pi) q[30];
rz(1.7947730354023008*pi) q[31];
rz(2.9546295053897005*pi) q[32];
rz(0.9847436935245*pi) q[33];
rz(2.0991066325856984*pi) q[34];
rz(3.9379487198099987*pi) q[35];
rz(3.1673378172734985*pi) q[36];
rz(2.613924494719999*pi) q[37];
rz(3.5276441183805005*pi) q[38];
rz(1.4023319481988992*pi) q[39];
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
