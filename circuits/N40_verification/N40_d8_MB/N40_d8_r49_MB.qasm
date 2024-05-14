OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.597178829134706*pi,1.9350261943334734*pi) q[0];
U1q(1.2540477668592*pi,0.11985067320293287*pi) q[1];
U1q(0.275291951177591*pi,0.459342716862104*pi) q[2];
U1q(0.448729167691763*pi,1.447426000373146*pi) q[3];
U1q(0.522476612970311*pi,1.42186454653621*pi) q[4];
U1q(0.67344866045904*pi,0.9328288610467399*pi) q[5];
U1q(0.710002665655442*pi,1.574539740121637*pi) q[6];
U1q(0.374475292846462*pi,0.381444978443033*pi) q[7];
U1q(0.835621156294005*pi,0.94525622066654*pi) q[8];
U1q(3.603984289617365*pi,1.0963575055868555*pi) q[9];
U1q(0.378592063207241*pi,0.153224265344696*pi) q[10];
U1q(0.648378654842739*pi,0.8594975008236001*pi) q[11];
U1q(1.61653722649176*pi,0.485562767268395*pi) q[12];
U1q(1.58569360602928*pi,1.6329720811459385*pi) q[13];
U1q(3.32068042649947*pi,1.3876903119040578*pi) q[14];
U1q(1.71496849469912*pi,1.6119812493200534*pi) q[15];
U1q(0.568712113668663*pi,0.118584565188773*pi) q[16];
U1q(0.627515781385036*pi,1.579920046093717*pi) q[17];
U1q(1.56194374445869*pi,1.603627241675647*pi) q[18];
U1q(0.5757826332668*pi,0.635341697067424*pi) q[19];
U1q(0.463961320759876*pi,1.669692082204647*pi) q[20];
U1q(0.478665809127582*pi,0.7718445585518099*pi) q[21];
U1q(0.198378275874367*pi,1.203864979135783*pi) q[22];
U1q(0.129902059370332*pi,0.9446622022923401*pi) q[23];
U1q(0.757630450460069*pi,1.25114203673656*pi) q[24];
U1q(0.397526735672721*pi,1.660643679437016*pi) q[25];
U1q(0.854615854808918*pi,1.567494439297715*pi) q[26];
U1q(1.27369461088951*pi,1.290649093298354*pi) q[27];
U1q(0.435900587311124*pi,1.211577655944775*pi) q[28];
U1q(1.54898099283578*pi,1.0253308470019946*pi) q[29];
U1q(1.97110529621755*pi,1.5728809576733123*pi) q[30];
U1q(0.534207428833027*pi,1.46533905198007*pi) q[31];
U1q(1.46255142284597*pi,0.3144282006654299*pi) q[32];
U1q(0.398097887957757*pi,0.0700302413309739*pi) q[33];
U1q(1.44662278353935*pi,0.8993518968301993*pi) q[34];
U1q(1.662221443255*pi,1.2648774679703965*pi) q[35];
U1q(0.680992764844982*pi,0.309915649925308*pi) q[36];
U1q(0.428312719253661*pi,1.763480268851159*pi) q[37];
U1q(3.692610181816319*pi,0.5777901795830477*pi) q[38];
U1q(3.521599005938776*pi,0.5436351712821209*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[1],q[39];
RZZ(0.5*pi) q[27],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[4],q[26];
RZZ(0.5*pi) q[31],q[5];
RZZ(0.5*pi) q[29],q[6];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[11],q[9];
RZZ(0.5*pi) q[19],q[10];
RZZ(0.5*pi) q[13],q[34];
RZZ(0.5*pi) q[16],q[15];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[37],q[24];
RZZ(0.5*pi) q[25],q[35];
RZZ(0.5*pi) q[32],q[33];
U1q(0.676680436881141*pi,0.47203766089569*pi) q[0];
U1q(0.62709453421774*pi,0.811056214541503*pi) q[1];
U1q(0.493982049197203*pi,1.50260103698219*pi) q[2];
U1q(0.135013569246447*pi,1.2316097357394802*pi) q[3];
U1q(0.681191176498744*pi,1.2844076052543691*pi) q[4];
U1q(0.388817658470169*pi,1.90183933523591*pi) q[5];
U1q(0.283541030666789*pi,0.9959817756987901*pi) q[6];
U1q(0.883272454183886*pi,0.08139768135922987*pi) q[7];
U1q(0.706598699310957*pi,0.10035741759711003*pi) q[8];
U1q(0.677151503809574*pi,0.49257298847636566*pi) q[9];
U1q(0.750530120536101*pi,1.8051205608766598*pi) q[10];
U1q(0.269069630293382*pi,0.7393492496655503*pi) q[11];
U1q(0.333423599230295*pi,1.5069259856852848*pi) q[12];
U1q(0.487277413038804*pi,1.6708242535115687*pi) q[13];
U1q(0.570595684525566*pi,1.9350772521362476*pi) q[14];
U1q(0.844392889450659*pi,0.28421878654110344*pi) q[15];
U1q(0.973087951588373*pi,0.27649572884038*pi) q[16];
U1q(0.814674018250198*pi,1.9333191458758199*pi) q[17];
U1q(0.352894692157914*pi,0.07643671194107693*pi) q[18];
U1q(0.470637177839984*pi,1.185293545769776*pi) q[19];
U1q(0.442724431035426*pi,0.99944995634293*pi) q[20];
U1q(0.610902340392023*pi,1.82639221237749*pi) q[21];
U1q(0.341112441692179*pi,1.64570937581597*pi) q[22];
U1q(0.7258508472309*pi,1.66583691286184*pi) q[23];
U1q(0.759223445345439*pi,0.95133106909601*pi) q[24];
U1q(0.0467965152472943*pi,0.022360908631060106*pi) q[25];
U1q(0.730710601986058*pi,0.5965906804133101*pi) q[26];
U1q(0.370223316771353*pi,0.9011034455909641*pi) q[27];
U1q(0.194230284801173*pi,1.1968832614099298*pi) q[28];
U1q(0.888325034523076*pi,0.8675270813302047*pi) q[29];
U1q(0.624607789980151*pi,1.924708802421292*pi) q[30];
U1q(0.166754950424479*pi,0.354280495290148*pi) q[31];
U1q(0.604814097923415*pi,0.3440963085986999*pi) q[32];
U1q(0.701939315182141*pi,0.7923734125065902*pi) q[33];
U1q(0.614970476096042*pi,1.7971144527186094*pi) q[34];
U1q(0.695392892641003*pi,1.9103782944514065*pi) q[35];
U1q(0.465386066404824*pi,0.48217502047601*pi) q[36];
U1q(0.288236505059188*pi,1.9615712391920401*pi) q[37];
U1q(0.689548399121497*pi,1.7608460700424626*pi) q[38];
U1q(0.117158351499645*pi,0.11128807883892122*pi) q[39];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[8],q[18];
RZZ(0.5*pi) q[36],q[9];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[12],q[39];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[17],q[26];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[21],q[34];
RZZ(0.5*pi) q[37],q[22];
RZZ(0.5*pi) q[24],q[29];
RZZ(0.5*pi) q[28],q[33];
RZZ(0.5*pi) q[32],q[35];
U1q(0.386763807338588*pi,0.4565840111762203*pi) q[0];
U1q(0.633768967523597*pi,0.618542816809323*pi) q[1];
U1q(0.852728443680261*pi,1.9060170443755897*pi) q[2];
U1q(0.176919285795664*pi,0.45332543636642963*pi) q[3];
U1q(0.481817716662355*pi,1.8574792528588802*pi) q[4];
U1q(0.583315835148419*pi,1.7919612121145603*pi) q[5];
U1q(0.767455568958925*pi,1.13948297012832*pi) q[6];
U1q(0.545044737384572*pi,0.04060692912081976*pi) q[7];
U1q(0.559442169810633*pi,1.32156703343379*pi) q[8];
U1q(0.71336175701834*pi,1.5202205288210253*pi) q[9];
U1q(0.0335681502688169*pi,0.12359843166040996*pi) q[10];
U1q(0.592703304777457*pi,0.44944910055815956*pi) q[11];
U1q(0.942375432846172*pi,1.3419701460572258*pi) q[12];
U1q(0.150751330963276*pi,0.3032149603936083*pi) q[13];
U1q(0.340939627768837*pi,0.7595819064856872*pi) q[14];
U1q(0.453719069252752*pi,1.1529755186709632*pi) q[15];
U1q(0.425337087897369*pi,1.07448048990934*pi) q[16];
U1q(0.392884725054595*pi,1.2413133615410104*pi) q[17];
U1q(0.344162699415989*pi,0.741057948676727*pi) q[18];
U1q(0.336459086124326*pi,1.98402375727082*pi) q[19];
U1q(0.280107129410169*pi,1.9907858381353902*pi) q[20];
U1q(0.356461469151968*pi,1.0925296617641198*pi) q[21];
U1q(0.563493846487732*pi,0.3876513336085896*pi) q[22];
U1q(0.48410128802111*pi,0.41328488915077966*pi) q[23];
U1q(0.13746959757868*pi,0.04528230907863007*pi) q[24];
U1q(0.462620062798828*pi,0.40556293871230986*pi) q[25];
U1q(0.18484677148157*pi,1.89037935666307*pi) q[26];
U1q(0.217048036566978*pi,1.945600995433594*pi) q[27];
U1q(0.788530792017658*pi,1.6876939035302803*pi) q[28];
U1q(0.205245101601914*pi,1.8112820528617242*pi) q[29];
U1q(0.75861731847051*pi,1.1807086655497816*pi) q[30];
U1q(0.609709760523627*pi,1.6203885595882*pi) q[31];
U1q(0.850139287182211*pi,0.07610531450960023*pi) q[32];
U1q(0.688132344887498*pi,1.7799793752584696*pi) q[33];
U1q(0.555794814954046*pi,1.8853820538878594*pi) q[34];
U1q(0.967952072338404*pi,1.4900434023573865*pi) q[35];
U1q(0.178295985822884*pi,0.6908069637716796*pi) q[36];
U1q(0.480378388449469*pi,0.21891506447667997*pi) q[37];
U1q(0.545560579198193*pi,1.2726870381029278*pi) q[38];
U1q(0.355442468323137*pi,0.9225840279473614*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[2],q[19];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[14],q[4];
RZZ(0.5*pi) q[5],q[26];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[8],q[36];
RZZ(0.5*pi) q[18],q[9];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[27],q[29];
RZZ(0.5*pi) q[28],q[39];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[32],q[37];
U1q(0.459324200697271*pi,1.1622406304631898*pi) q[0];
U1q(0.917364988204449*pi,1.5232122936837529*pi) q[1];
U1q(0.0217651828745698*pi,1.2222463201326903*pi) q[2];
U1q(0.549293589053603*pi,0.025578978884319703*pi) q[3];
U1q(0.826914484494133*pi,0.82082728157354*pi) q[4];
U1q(0.208421532209969*pi,0.9917576979724201*pi) q[5];
U1q(0.608776627472385*pi,0.5232913631668898*pi) q[6];
U1q(0.737508308537749*pi,0.95785540678812*pi) q[7];
U1q(0.630200330575722*pi,1.7092875023398992*pi) q[8];
U1q(0.833034151004236*pi,1.3222530145094051*pi) q[9];
U1q(0.804334446745786*pi,0.90679278453565*pi) q[10];
U1q(0.23507549047218*pi,0.3639850782991001*pi) q[11];
U1q(0.726900578815305*pi,1.878169006792115*pi) q[12];
U1q(0.7320454785437*pi,0.6728519481347384*pi) q[13];
U1q(0.641589560716809*pi,1.0127206011374774*pi) q[14];
U1q(0.324628396414973*pi,1.0737636886990138*pi) q[15];
U1q(0.373955421629678*pi,0.18903833825608007*pi) q[16];
U1q(0.53137079346368*pi,0.6117051558372504*pi) q[17];
U1q(0.361909771980301*pi,1.681774225341396*pi) q[18];
U1q(0.3417418921812*pi,1.94449516173386*pi) q[19];
U1q(0.288967061974933*pi,0.6832294254906701*pi) q[20];
U1q(0.675374319034032*pi,0.39648688537898025*pi) q[21];
U1q(0.401318486894022*pi,1.6947469143275509*pi) q[22];
U1q(0.736354870954053*pi,0.11127793900416005*pi) q[23];
U1q(0.973944291433367*pi,0.8182754486135*pi) q[24];
U1q(0.498531385274528*pi,0.6254386747240499*pi) q[25];
U1q(0.39046194217404*pi,1.7517905915107104*pi) q[26];
U1q(0.249859819046872*pi,0.2597704224214743*pi) q[27];
U1q(0.412780065103046*pi,0.6321721604094304*pi) q[28];
U1q(0.657238920828856*pi,1.0984998308310452*pi) q[29];
U1q(0.802935024761142*pi,1.9723579980462915*pi) q[30];
U1q(0.498246958088301*pi,1.5155929293743*pi) q[31];
U1q(0.530056749446794*pi,1.6195239037597595*pi) q[32];
U1q(0.381039138525786*pi,0.5206594672950704*pi) q[33];
U1q(0.538044589040099*pi,1.6233685728957798*pi) q[34];
U1q(0.609187113546812*pi,1.5310699308418156*pi) q[35];
U1q(0.204989671826043*pi,0.24192612790174994*pi) q[36];
U1q(0.785928435821177*pi,1.7911578745878192*pi) q[37];
U1q(0.243434663741914*pi,1.100259166967037*pi) q[38];
U1q(0.581193438609545*pi,1.6617667931138307*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[29],q[2];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[4],q[25];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[8],q[28];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[32],q[10];
RZZ(0.5*pi) q[36],q[11];
RZZ(0.5*pi) q[37],q[15];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[18],q[22];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[24],q[39];
RZZ(0.5*pi) q[31],q[26];
RZZ(0.5*pi) q[35],q[27];
U1q(0.468383067032232*pi,1.6148384326943894*pi) q[0];
U1q(0.252098390026834*pi,0.10097107100113334*pi) q[1];
U1q(0.395108917496881*pi,1.8276326311758702*pi) q[2];
U1q(0.687337713717797*pi,1.3643329483966404*pi) q[3];
U1q(0.554748601785593*pi,0.5069284570469703*pi) q[4];
U1q(0.470948024655848*pi,0.9040544707071998*pi) q[5];
U1q(0.430876730207473*pi,0.09158970526434018*pi) q[6];
U1q(0.73493125959708*pi,1.2370384081477592*pi) q[7];
U1q(0.25376393966391*pi,1.8420844068846005*pi) q[8];
U1q(0.675394883373642*pi,1.059571183033155*pi) q[9];
U1q(0.316073706990919*pi,1.8213989201586998*pi) q[10];
U1q(0.622313088422497*pi,0.8554966296131994*pi) q[11];
U1q(0.828757518733611*pi,1.0492677747920442*pi) q[12];
U1q(0.348866689034137*pi,0.31972434886319867*pi) q[13];
U1q(0.686615753122574*pi,0.6571256243014876*pi) q[14];
U1q(0.795804450667867*pi,0.8753076083112035*pi) q[15];
U1q(0.676411808892086*pi,0.9748201499441809*pi) q[16];
U1q(0.721955150067251*pi,0.3568721918065396*pi) q[17];
U1q(0.408852384613224*pi,1.3198777434005269*pi) q[18];
U1q(0.723387752987437*pi,0.5856576144238304*pi) q[19];
U1q(0.71459406655277*pi,0.2763309226623196*pi) q[20];
U1q(0.540525429293045*pi,1.0844473112201296*pi) q[21];
U1q(0.813833950967719*pi,1.6857879239979*pi) q[22];
U1q(0.607414418747056*pi,0.32329444219709025*pi) q[23];
U1q(0.335829920541834*pi,0.9021820358318298*pi) q[24];
U1q(0.599789874158921*pi,1.6289492570695296*pi) q[25];
U1q(0.312965748368779*pi,0.5375217706043696*pi) q[26];
U1q(0.456536210576645*pi,1.198917107538234*pi) q[27];
U1q(0.256381471210892*pi,0.4611202123794005*pi) q[28];
U1q(0.296248294590349*pi,0.22349449764049467*pi) q[29];
U1q(0.676681883712797*pi,1.547661668475012*pi) q[30];
U1q(0.393348741368086*pi,0.037913283986860336*pi) q[31];
U1q(0.204009105577386*pi,1.4920512875923304*pi) q[32];
U1q(0.343103924995753*pi,1.9421814706374398*pi) q[33];
U1q(0.274077083326427*pi,0.2591580201242998*pi) q[34];
U1q(0.3440443235818*pi,0.6990703035954677*pi) q[35];
U1q(0.744895232239169*pi,1.5744539265865996*pi) q[36];
U1q(0.30801191618576*pi,0.5356525045002005*pi) q[37];
U1q(0.54310689945748*pi,1.885692134937747*pi) q[38];
U1q(0.386314644853818*pi,0.5826232113398913*pi) q[39];
rz(0.9252003271370697*pi) q[0];
rz(1.176110627083327*pi) q[1];
rz(3.9401650778081203*pi) q[2];
rz(1.95837672905966*pi) q[3];
rz(0.036924621716000416*pi) q[4];
rz(2.0822933421094003*pi) q[5];
rz(0.07700637507820929*pi) q[6];
rz(0.03430538635053004*pi) q[7];
rz(1.1051418729412994*pi) q[8];
rz(2.684184833825725*pi) q[9];
rz(2.0245080018683996*pi) q[10];
rz(1.822583911013*pi) q[11];
rz(1.5149377337371064*pi) q[12];
rz(2.074121841343471*pi) q[13];
rz(3.811783897832293*pi) q[14];
rz(1.243818996167576*pi) q[15];
rz(1.3800675290970403*pi) q[16];
rz(2.5325843729804003*pi) q[17];
rz(0.9445992379500332*pi) q[18];
rz(2.473425578796901*pi) q[19];
rz(3.0182749560040403*pi) q[20];
rz(0.5887688258083692*pi) q[21];
rz(2.9956488264409007*pi) q[22];
rz(2.6697200804486005*pi) q[23];
rz(2.67089729598389*pi) q[24];
rz(1.7861606361589804*pi) q[25];
rz(2.73637075845585*pi) q[26];
rz(1.1311878677578058*pi) q[27];
rz(3.0423738662071003*pi) q[28];
rz(0.3297959631219065*pi) q[29];
rz(3.579724670688888*pi) q[30];
rz(2.8823349372930602*pi) q[31];
rz(2.1280802588048697*pi) q[32];
rz(3.8156263713265997*pi) q[33];
rz(3.3678535111441006*pi) q[34];
rz(0.5083948701808936*pi) q[35];
rz(0.1378877837735395*pi) q[36];
rz(3.7694514443721*pi) q[37];
rz(1.479669155267052*pi) q[38];
rz(0.01737540585634889*pi) q[39];
U1q(0.468383067032232*pi,1.540038759831459*pi) q[0];
U1q(1.25209839002683*pi,0.277081698084474*pi) q[1];
U1q(1.39510891749688*pi,0.767797708984*pi) q[2];
U1q(1.6873377137178*pi,0.322709677456304*pi) q[3];
U1q(1.55474860178559*pi,1.543853078762969*pi) q[4];
U1q(1.47094802465585*pi,1.9863478128166012*pi) q[5];
U1q(1.43087673020747*pi,1.1685960803425481*pi) q[6];
U1q(0.73493125959708*pi,0.271343794498293*pi) q[7];
U1q(1.25376393966391*pi,1.9472262798259*pi) q[8];
U1q(0.675394883373642*pi,0.743756016858837*pi) q[9];
U1q(0.316073706990919*pi,0.8459069220270401*pi) q[10];
U1q(1.6223130884225*pi,1.678080540626222*pi) q[11];
U1q(0.828757518733611*pi,1.564205508529106*pi) q[12];
U1q(0.348866689034137*pi,1.393846190206677*pi) q[13];
U1q(1.68661575312257*pi,1.46890952213378*pi) q[14];
U1q(1.79580445066787*pi,1.11912660447878*pi) q[15];
U1q(0.676411808892086*pi,1.35488767904122*pi) q[16];
U1q(0.721955150067251*pi,1.889456564786916*pi) q[17];
U1q(1.40885238461322*pi,1.26447698135056*pi) q[18];
U1q(1.72338775298744*pi,0.0590831932207631*pi) q[19];
U1q(0.71459406655277*pi,0.294605878666368*pi) q[20];
U1q(1.54052542929305*pi,0.6732161370285099*pi) q[21];
U1q(3.813833950967719*pi,1.681436750438788*pi) q[22];
U1q(0.607414418747056*pi,1.9930145226457137*pi) q[23];
U1q(1.33582992054183*pi,0.573079331815723*pi) q[24];
U1q(1.59978987415892*pi,0.415109893228513*pi) q[25];
U1q(0.312965748368779*pi,0.273892529060219*pi) q[26];
U1q(3.456536210576645*pi,1.330104975296043*pi) q[27];
U1q(0.256381471210892*pi,0.503494078586566*pi) q[28];
U1q(0.296248294590349*pi,1.55329046076241*pi) q[29];
U1q(0.676681883712797*pi,0.127386339163898*pi) q[30];
U1q(0.393348741368086*pi,1.9202482212799274*pi) q[31];
U1q(1.20400910557739*pi,0.620131546397178*pi) q[32];
U1q(1.34310392499575*pi,0.757807841964042*pi) q[33];
U1q(0.274077083326427*pi,0.627011531268402*pi) q[34];
U1q(1.3440443235818*pi,0.207465173776363*pi) q[35];
U1q(0.744895232239169*pi,0.712341710360149*pi) q[36];
U1q(0.30801191618576*pi,1.3051039488722709*pi) q[37];
U1q(0.54310689945748*pi,0.365361290204782*pi) q[38];
U1q(0.386314644853818*pi,1.599998617196243*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[29],q[2];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[4],q[25];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[8],q[28];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[32],q[10];
RZZ(0.5*pi) q[36],q[11];
RZZ(0.5*pi) q[37],q[15];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[18],q[22];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[24],q[39];
RZZ(0.5*pi) q[31],q[26];
RZZ(0.5*pi) q[35],q[27];
U1q(0.459324200697271*pi,0.08744095760024995*pi) q[0];
U1q(3.082635011795552*pi,1.8548404754018692*pi) q[1];
U1q(1.02176518287457*pi,1.3731840200271865*pi) q[2];
U1q(3.450706410946397*pi,1.6614636469686201*pi) q[3];
U1q(1.82691448449413*pi,0.22995425423639126*pi) q[4];
U1q(1.20842153220997*pi,1.898644585551386*pi) q[5];
U1q(1.60877662747239*pi,1.7368944224399931*pi) q[6];
U1q(0.737508308537749*pi,0.99216079313865*pi) q[7];
U1q(3.369799669424278*pi,1.0800231843705292*pi) q[8];
U1q(0.833034151004236*pi,0.00643784833513128*pi) q[9];
U1q(0.804334446745786*pi,1.9313007864040497*pi) q[10];
U1q(3.23507549047218*pi,0.16959209194030955*pi) q[11];
U1q(1.72690057881531*pi,0.39310674052916994*pi) q[12];
U1q(1.7320454785437*pi,0.74697378947821*pi) q[13];
U1q(1.64158956071681*pi,0.1133145452977864*pi) q[14];
U1q(1.32462839641497*pi,0.9206705240909765*pi) q[15];
U1q(1.37395542162968*pi,1.569105867353122*pi) q[16];
U1q(0.53137079346368*pi,0.14428952881762003*pi) q[17];
U1q(3.638090228019698*pi,0.9025804994096839*pi) q[18];
U1q(3.3417418921812*pi,1.7002456459107385*pi) q[19];
U1q(0.288967061974933*pi,1.701504381494716*pi) q[20];
U1q(3.324625680965967*pi,0.36117656286966593*pi) q[21];
U1q(3.598681513105978*pi,1.6724777601091572*pi) q[22];
U1q(1.73635487095405*pi,0.7809980194527899*pi) q[23];
U1q(3.026055708566633*pi,1.6569859190340548*pi) q[24];
U1q(1.49853138527453*pi,0.41862047557399673*pi) q[25];
U1q(0.39046194217404*pi,0.48816134996656*pi) q[26];
U1q(1.24985981904687*pi,1.2692516604128032*pi) q[27];
U1q(1.41278006510305*pi,0.6745460266165699*pi) q[28];
U1q(0.657238920828856*pi,0.4282957939529699*pi) q[29];
U1q(1.80293502476114*pi,0.5520826687352001*pi) q[30];
U1q(0.498246958088301*pi,0.397927866667361*pi) q[31];
U1q(3.469943250553206*pi,1.4926589302297466*pi) q[32];
U1q(3.618960861474214*pi,1.1793298453064165*pi) q[33];
U1q(1.5380445890401*pi,0.991222084039885*pi) q[34];
U1q(3.390812886453188*pi,1.3754655465300127*pi) q[35];
U1q(1.20498967182604*pi,1.3798139116752899*pi) q[36];
U1q(0.785928435821177*pi,0.56060931895989*pi) q[37];
U1q(1.24343466374191*pi,1.5799283222340699*pi) q[38];
U1q(0.581193438609545*pi,0.67914219897018*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[2],q[19];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[14],q[4];
RZZ(0.5*pi) q[5],q[26];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[8],q[36];
RZZ(0.5*pi) q[18],q[9];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[27],q[29];
RZZ(0.5*pi) q[28],q[39];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[32],q[37];
U1q(0.386763807338588*pi,0.38178433831328995*pi) q[0];
U1q(3.366231032476403*pi,1.759509952276296*pi) q[1];
U1q(0.852728443680261*pi,1.0569547442700766*pi) q[2];
U1q(1.17691928579566*pi,1.2337171894865189*pi) q[3];
U1q(1.48181771666235*pi,0.2666062255217223*pi) q[4];
U1q(1.58331583514842*pi,1.6988480996935278*pi) q[5];
U1q(1.76745556895893*pi,0.3530860294014153*pi) q[6];
U1q(0.545044737384572*pi,1.074912315471346*pi) q[7];
U1q(1.55944216981063*pi,1.4677436532766843*pi) q[8];
U1q(0.71336175701834*pi,0.2044053626467499*pi) q[9];
U1q(0.0335681502688169*pi,1.1481064335288096*pi) q[10];
U1q(1.59270330477746*pi,0.2550561141993575*pi) q[11];
U1q(1.94237543284617*pi,1.9293056012640548*pi) q[12];
U1q(3.849248669036722*pi,1.1166107772193326*pi) q[13];
U1q(0.340939627768837*pi,0.8601758506459966*pi) q[14];
U1q(1.45371906925275*pi,1.9998823540629285*pi) q[15];
U1q(3.4253370878973692*pi,1.6836637156998604*pi) q[16];
U1q(1.3928847250546*pi,1.7738977345213804*pi) q[17];
U1q(3.655837300584011*pi,1.8432967760743548*pi) q[18];
U1q(0.336459086124326*pi,0.7397742414476955*pi) q[19];
U1q(1.28010712941017*pi,0.009060794139430062*pi) q[20];
U1q(3.643538530848032*pi,1.6651337864845193*pi) q[21];
U1q(1.56349384648773*pi,0.9795733408281155*pi) q[22];
U1q(1.48410128802111*pi,1.4789910693061747*pi) q[23];
U1q(3.86253040242132*pi,0.4299790585689284*pi) q[24];
U1q(0.462620062798828*pi,1.1987447395622677*pi) q[25];
U1q(0.18484677148157*pi,1.62675011511892*pi) q[26];
U1q(0.217048036566978*pi,1.9550822334249331*pi) q[27];
U1q(3.788530792017659*pi,1.6190242834957247*pi) q[28];
U1q(0.205245101601914*pi,0.14107801598365022*pi) q[29];
U1q(3.758617318470511*pi,0.34373200123171355*pi) q[30];
U1q(0.609709760523627*pi,0.5027234968812699*pi) q[31];
U1q(3.14986071281779*pi,1.0360775194799157*pi) q[32];
U1q(1.6881323448875*pi,0.9200099373430128*pi) q[33];
U1q(1.55579481495405*pi,1.729208603047812*pi) q[34];
U1q(1.9679520723384*pi,1.4164920750144345*pi) q[35];
U1q(3.821704014177116*pi,0.9309330758053553*pi) q[36];
U1q(1.48037838844947*pi,0.9883665088487401*pi) q[37];
U1q(1.54556057919819*pi,1.407500451098175*pi) q[38];
U1q(1.35544246832314*pi,1.9399594338037103*pi) q[39];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[8],q[18];
RZZ(0.5*pi) q[36],q[9];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[12],q[39];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[17],q[26];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[21],q[34];
RZZ(0.5*pi) q[37],q[22];
RZZ(0.5*pi) q[24],q[29];
RZZ(0.5*pi) q[28],q[33];
RZZ(0.5*pi) q[32],q[35];
U1q(0.676680436881141*pi,0.39723798803275034*pi) q[0];
U1q(1.62709453421774*pi,0.5669965545441218*pi) q[1];
U1q(0.493982049197203*pi,1.6535387368766865*pi) q[2];
U1q(0.135013569246447*pi,1.012001488859572*pi) q[3];
U1q(1.68119117649874*pi,0.8396778731262304*pi) q[4];
U1q(3.611182341529831*pi,0.5889699765721699*pi) q[5];
U1q(3.716458969333211*pi,1.4965872238309466*pi) q[6];
U1q(0.883272454183886*pi,1.11570306770976*pi) q[7];
U1q(1.70659869931096*pi,0.24653403744000402*pi) q[8];
U1q(0.677151503809574*pi,1.1767578223020898*pi) q[9];
U1q(0.750530120536101*pi,0.8296285627450608*pi) q[10];
U1q(3.269069630293381*pi,1.9651559650919683*pi) q[11];
U1q(0.333423599230295*pi,0.09426144089211519*pi) q[12];
U1q(3.512722586961196*pi,1.7490014841013746*pi) q[13];
U1q(1.57059568452557*pi,0.03567119629654636*pi) q[14];
U1q(3.155607110549342*pi,1.8686390861927886*pi) q[15];
U1q(1.97308795158837*pi,1.8856789546309103*pi) q[16];
U1q(3.185325981749802*pi,0.08189195018657314*pi) q[17];
U1q(3.647105307842086*pi,0.5079180128100069*pi) q[18];
U1q(0.470637177839984*pi,0.9410440299466556*pi) q[19];
U1q(3.557275568964574*pi,0.0003966759318916324*pi) q[20];
U1q(3.389097659607977*pi,0.931271235871157*pi) q[21];
U1q(1.34111244169218*pi,0.23763138303548548*pi) q[22];
U1q(1.7258508472309*pi,0.7315430930172377*pi) q[23];
U1q(3.240776554654561*pi,1.5239302985515497*pi) q[24];
U1q(0.0467965152472943*pi,1.8155427094810168*pi) q[25];
U1q(0.730710601986058*pi,0.33296143886915974*pi) q[26];
U1q(1.37022331677135*pi,0.9105846835823037*pi) q[27];
U1q(0.194230284801173*pi,0.1282136413753845*pi) q[28];
U1q(0.888325034523076*pi,1.1973230444521201*pi) q[29];
U1q(1.62460778998015*pi,1.0877321381032234*pi) q[30];
U1q(0.166754950424479*pi,0.23661543258321016*pi) q[31];
U1q(3.395185902076585*pi,0.7680865253908156*pi) q[32];
U1q(1.70193931518214*pi,0.9324039745911288*pi) q[33];
U1q(1.61497047609604*pi,1.6409410018785646*pi) q[34];
U1q(0.695392892641003*pi,1.8368269671084585*pi) q[35];
U1q(1.46538606640482*pi,0.13956501910102648*pi) q[36];
U1q(1.28823650505919*pi,1.2457103341333755*pi) q[37];
U1q(1.6895483991215*pi,1.8956594830377111*pi) q[38];
U1q(3.882841648500353*pi,1.751255382912139*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[1],q[39];
RZZ(0.5*pi) q[27],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[4],q[26];
RZZ(0.5*pi) q[31],q[5];
RZZ(0.5*pi) q[29],q[6];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[11],q[9];
RZZ(0.5*pi) q[19],q[10];
RZZ(0.5*pi) q[13],q[34];
RZZ(0.5*pi) q[16],q[15];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[37],q[24];
RZZ(0.5*pi) q[25],q[35];
RZZ(0.5*pi) q[32],q[33];
U1q(0.597178829134706*pi,1.8602265214705396*pi) q[0];
U1q(0.254047766859202*pi,1.8757910132055557*pi) q[1];
U1q(0.275291951177591*pi,0.6102804167565967*pi) q[2];
U1q(0.448729167691763*pi,0.22781775349324196*pi) q[3];
U1q(0.522476612970311*pi,1.97713481440808*pi) q[4];
U1q(1.67344866045904*pi,0.5579804507613524*pi) q[5];
U1q(1.71000266565544*pi,1.9180292594081019*pi) q[6];
U1q(0.374475292846462*pi,1.4157503647935599*pi) q[7];
U1q(1.83562115629401*pi,0.401635234370576*pi) q[8];
U1q(0.603984289617365*pi,0.7805423394125803*pi) q[9];
U1q(0.378592063207241*pi,0.17773226721308966*pi) q[10];
U1q(0.648378654842739*pi,1.0853042162500084*pi) q[11];
U1q(0.616537226491756*pi,0.07289822247522526*pi) q[12];
U1q(1.58569360602928*pi,1.7868536564670028*pi) q[13];
U1q(3.32068042649947*pi,0.5830581365287366*pi) q[14];
U1q(3.714968494699124*pi,0.5408766234138378*pi) q[15];
U1q(1.56871211366866*pi,1.043590118282519*pi) q[16];
U1q(1.62751578138504*pi,0.4352910499686793*pi) q[17];
U1q(3.561943744458693*pi,1.9807274830754404*pi) q[18];
U1q(0.5757826332668*pi,1.3910921812443053*pi) q[19];
U1q(1.46396132075988*pi,0.33015455007017325*pi) q[20];
U1q(3.478665809127582*pi,0.985818889696839*pi) q[21];
U1q(1.19837827587437*pi,1.6794757797156694*pi) q[22];
U1q(1.12990205937033*pi,0.4527178035867405*pi) q[23];
U1q(1.75763045046007*pi,0.22411933091099168*pi) q[24];
U1q(0.397526735672721*pi,0.4538254802869668*pi) q[25];
U1q(0.854615854808918*pi,0.30386519775356025*pi) q[26];
U1q(1.27369461088951*pi,0.5210390358749146*pi) q[27];
U1q(0.435900587311124*pi,0.14290803591022438*pi) q[28];
U1q(0.548980992835785*pi,1.3551268101239202*pi) q[29];
U1q(1.97110529621755*pi,0.4395599828512058*pi) q[30];
U1q(0.534207428833027*pi,1.3476739892731402*pi) q[31];
U1q(1.46255142284597*pi,0.7977546333240841*pi) q[32];
U1q(3.398097887957757*pi,0.6547471457667462*pi) q[33];
U1q(1.44662278353935*pi,1.538703557766977*pi) q[34];
U1q(0.662221443254996*pi,0.19132614062744469*pi) q[35];
U1q(0.680992764844982*pi,1.9673056485503264*pi) q[36];
U1q(0.428312719253661*pi,1.0476193637924958*pi) q[37];
U1q(1.69261018181632*pi,0.07871537349713065*pi) q[38];
U1q(1.52159900593878*pi,0.318908290468948*pi) q[39];
rz(2.1397734785294604*pi) q[0];
rz(2.1242089867944443*pi) q[1];
rz(1.3897195832434033*pi) q[2];
rz(1.772182246506758*pi) q[3];
rz(2.02286518559192*pi) q[4];
rz(3.4420195492386476*pi) q[5];
rz(2.081970740591898*pi) q[6];
rz(0.5842496352064401*pi) q[7];
rz(3.598364765629424*pi) q[8];
rz(1.2194576605874197*pi) q[9];
rz(1.8222677327869103*pi) q[10];
rz(2.9146957837499916*pi) q[11];
rz(1.9271017775247747*pi) q[12];
rz(0.21314634353299725*pi) q[13];
rz(1.4169418634712634*pi) q[14];
rz(3.459123376586162*pi) q[15];
rz(2.956409881717481*pi) q[16];
rz(3.5647089500313207*pi) q[17];
rz(2.0192725169245596*pi) q[18];
rz(2.6089078187556947*pi) q[19];
rz(1.6698454499298268*pi) q[20];
rz(1.014181110303161*pi) q[21];
rz(2.3205242202843306*pi) q[22];
rz(3.5472821964132595*pi) q[23];
rz(1.7758806690890083*pi) q[24];
rz(3.546174519713033*pi) q[25];
rz(3.6961348022464398*pi) q[26];
rz(3.4789609641250854*pi) q[27];
rz(1.8570919640897756*pi) q[28];
rz(2.6448731898760798*pi) q[29];
rz(3.560440017148794*pi) q[30];
rz(0.6523260107268598*pi) q[31];
rz(1.2022453666759159*pi) q[32];
rz(3.345252854233254*pi) q[33];
rz(0.4612964422330229*pi) q[34];
rz(1.8086738593725553*pi) q[35];
rz(2.0326943514496736*pi) q[36];
rz(0.9523806362075042*pi) q[37];
rz(3.921284626502869*pi) q[38];
rz(1.681091709531052*pi) q[39];
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