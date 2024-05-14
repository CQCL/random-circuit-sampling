OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.414351001003648*pi,0.366533958502151*pi) q[0];
U1q(0.373363929942742*pi,0.257029587346365*pi) q[1];
U1q(0.2607925585479*pi,0.686579711490573*pi) q[2];
U1q(0.332489792178867*pi,1.235152660852059*pi) q[3];
U1q(0.402419421169693*pi,1.48518920675525*pi) q[4];
U1q(0.367774119471411*pi,0.737034922348884*pi) q[5];
U1q(0.632993672279141*pi,1.26931449792464*pi) q[6];
U1q(0.757918817440806*pi,0.341212280137097*pi) q[7];
U1q(0.53808566290886*pi,1.9507909332610875*pi) q[8];
U1q(0.322343886871517*pi,0.358225682431816*pi) q[9];
U1q(0.828850487485127*pi,0.570986170821243*pi) q[10];
U1q(0.657218702536406*pi,1.3702013291268371*pi) q[11];
U1q(0.543971468631741*pi,0.69584744135301*pi) q[12];
U1q(0.32575911486649*pi,1.588112411942184*pi) q[13];
U1q(0.17626117130536*pi,0.3880070114307*pi) q[14];
U1q(0.30165882623524*pi,1.622039414862994*pi) q[15];
U1q(0.543283742061329*pi,0.102026983451*pi) q[16];
U1q(0.429120519678496*pi,0.411576620421775*pi) q[17];
U1q(0.515824463578726*pi,0.0232520179155875*pi) q[18];
U1q(0.35257501465799*pi,0.511028208475031*pi) q[19];
U1q(0.28539129097974*pi,0.4406844841028901*pi) q[20];
U1q(0.342737534616512*pi,1.137948071827429*pi) q[21];
U1q(0.697215102237487*pi,1.04238091678825*pi) q[22];
U1q(0.904689041519464*pi,0.376702242680197*pi) q[23];
U1q(0.230132181783913*pi,1.757397890113902*pi) q[24];
U1q(0.347804861323972*pi,0.87340720830779*pi) q[25];
U1q(0.797118697355668*pi,1.812208208786465*pi) q[26];
U1q(0.820808487623823*pi,1.7864932656481929*pi) q[27];
U1q(0.0385080383005557*pi,1.20523695653194*pi) q[28];
U1q(0.254502763514247*pi,0.64708693798608*pi) q[29];
U1q(0.857538287782018*pi,1.981597023352377*pi) q[30];
U1q(0.821858914524041*pi,0.470262014893856*pi) q[31];
U1q(0.118003527352666*pi,1.445608122923041*pi) q[32];
U1q(0.595958790320456*pi,1.890486178068989*pi) q[33];
U1q(0.692444438824605*pi,0.285043267846907*pi) q[34];
U1q(0.369511339864207*pi,0.707761349677028*pi) q[35];
U1q(0.362480035906797*pi,0.5688778355364199*pi) q[36];
U1q(0.465480163141629*pi,0.618798100798811*pi) q[37];
U1q(0.408046799646481*pi,0.32561970244174*pi) q[38];
U1q(0.511137717351163*pi,1.9598950636293906*pi) q[39];
RZZ(0.5*pi) q[7],q[0];
RZZ(0.5*pi) q[8],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[4],q[3];
RZZ(0.5*pi) q[11],q[5];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[24];
RZZ(0.5*pi) q[29],q[14];
RZZ(0.5*pi) q[15],q[35];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[18],q[39];
RZZ(0.5*pi) q[21],q[38];
RZZ(0.5*pi) q[23],q[31];
RZZ(0.5*pi) q[27],q[25];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[37],q[33];
U1q(0.612730420656934*pi,1.92028451081815*pi) q[0];
U1q(0.665415309249831*pi,0.6101181849959301*pi) q[1];
U1q(0.493165412135499*pi,0.9188734645402299*pi) q[2];
U1q(0.661891209859056*pi,0.2507251536952899*pi) q[3];
U1q(0.514432920876556*pi,0.4658348928542799*pi) q[4];
U1q(0.272641246137321*pi,0.9337911659857401*pi) q[5];
U1q(0.113156062781071*pi,1.067262351831022*pi) q[6];
U1q(0.819507987665033*pi,1.2423924671046*pi) q[7];
U1q(0.767342538074199*pi,1.431793727878224*pi) q[8];
U1q(0.362817004845348*pi,1.0309353755371609*pi) q[9];
U1q(0.778043378333672*pi,1.913722037787863*pi) q[10];
U1q(0.292126262567063*pi,0.06935543492016993*pi) q[11];
U1q(0.202300371282257*pi,1.5908216722859496*pi) q[12];
U1q(0.32330913032475*pi,0.7689821535160499*pi) q[13];
U1q(0.788008687599741*pi,0.69252699194351*pi) q[14];
U1q(0.28614752577115*pi,0.5180162763865199*pi) q[15];
U1q(0.671691083914105*pi,1.5657542000087799*pi) q[16];
U1q(0.358809257755325*pi,1.093788185872425*pi) q[17];
U1q(0.689088117229658*pi,1.203865262238631*pi) q[18];
U1q(0.451526418500278*pi,0.3102852337409501*pi) q[19];
U1q(0.121161955278278*pi,1.061917528756*pi) q[20];
U1q(0.564142333770656*pi,0.40081677146778993*pi) q[21];
U1q(0.213033883039732*pi,0.305848301878084*pi) q[22];
U1q(0.508031784155167*pi,0.8573769605519701*pi) q[23];
U1q(0.58344150027603*pi,0.68928229097844*pi) q[24];
U1q(0.589472565597237*pi,1.0127636238894961*pi) q[25];
U1q(0.82781062572884*pi,0.6470769896876898*pi) q[26];
U1q(0.22752392872127*pi,0.19639293645983003*pi) q[27];
U1q(0.881286274816027*pi,0.6700955173376699*pi) q[28];
U1q(0.623103194123442*pi,0.9300313247572101*pi) q[29];
U1q(0.0808694305258244*pi,0.5521172318464398*pi) q[30];
U1q(0.590132830445601*pi,1.9071264815652502*pi) q[31];
U1q(0.359243373451627*pi,1.29591575630749*pi) q[32];
U1q(0.507540059251987*pi,1.0287526565716099*pi) q[33];
U1q(0.651851916590106*pi,1.2619300580085309*pi) q[34];
U1q(0.51250230228281*pi,0.8273924880939001*pi) q[35];
U1q(0.737874328690064*pi,1.11749865893379*pi) q[36];
U1q(0.912454390340339*pi,0.83118124496889*pi) q[37];
U1q(0.846951541774666*pi,0.1586798778979801*pi) q[38];
U1q(0.676149190970108*pi,0.9058391003349799*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[19],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[4],q[25];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[8],q[35];
RZZ(0.5*pi) q[22],q[10];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[17],q[36];
RZZ(0.5*pi) q[18],q[24];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[23],q[21];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[30],q[37];
RZZ(0.5*pi) q[32],q[31];
U1q(0.62549804336288*pi,0.70422906530978*pi) q[0];
U1q(0.919326238120971*pi,1.9444461310937102*pi) q[1];
U1q(0.263956645488296*pi,0.9076299812495701*pi) q[2];
U1q(0.200988723116212*pi,1.01264371972914*pi) q[3];
U1q(0.60330656530953*pi,0.1711135256479399*pi) q[4];
U1q(0.440207413264439*pi,1.4465911242072496*pi) q[5];
U1q(0.546079938462968*pi,0.13993952966155*pi) q[6];
U1q(0.791711506488796*pi,1.3476618916596896*pi) q[7];
U1q(0.374119049235911*pi,1.46046399392202*pi) q[8];
U1q(0.81647178507505*pi,1.4740469323278003*pi) q[9];
U1q(0.65084396398731*pi,0.13745342301587993*pi) q[10];
U1q(0.701677470201163*pi,0.04756522101796001*pi) q[11];
U1q(0.676376986375609*pi,0.36973291314139*pi) q[12];
U1q(0.716878321048768*pi,0.6967267604729401*pi) q[13];
U1q(0.782491434567081*pi,1.0857723053940398*pi) q[14];
U1q(0.219110225008068*pi,0.33759569190443006*pi) q[15];
U1q(0.532796455729206*pi,1.8124071212597697*pi) q[16];
U1q(0.23480188669038*pi,0.9591204781308802*pi) q[17];
U1q(0.732027473394617*pi,0.5957822578464298*pi) q[18];
U1q(0.652708177522464*pi,0.3800190736315199*pi) q[19];
U1q(0.399270418656781*pi,0.08198249715002959*pi) q[20];
U1q(0.34168381524327*pi,0.5981581229438699*pi) q[21];
U1q(0.630217772407086*pi,1.80423020179134*pi) q[22];
U1q(0.22461502181411*pi,0.4455041313508499*pi) q[23];
U1q(0.18384178665801*pi,0.15287466438975983*pi) q[24];
U1q(0.372539386514907*pi,0.52556447669566*pi) q[25];
U1q(0.185548169093873*pi,0.21886035936381987*pi) q[26];
U1q(0.441463794726643*pi,1.2602195985054898*pi) q[27];
U1q(0.529761922022858*pi,1.4798919363681096*pi) q[28];
U1q(0.555828715544683*pi,1.6074137456630702*pi) q[29];
U1q(0.364204431028141*pi,1.1726282591704704*pi) q[30];
U1q(0.491422060675862*pi,0.05372666699317996*pi) q[31];
U1q(0.551414537074905*pi,0.3097829180483096*pi) q[32];
U1q(0.270842386595909*pi,1.8522360466795798*pi) q[33];
U1q(0.325611831479865*pi,1.7370905728008998*pi) q[34];
U1q(0.566241025188841*pi,0.5868489853491798*pi) q[35];
U1q(0.770253136143628*pi,1.6956868907281297*pi) q[36];
U1q(0.254534096196691*pi,1.18580762522447*pi) q[37];
U1q(0.460471704580989*pi,1.6080094603522896*pi) q[38];
U1q(0.818103627378998*pi,1.3753232957864898*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[29],q[5];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[16],q[7];
RZZ(0.5*pi) q[8],q[39];
RZZ(0.5*pi) q[9],q[20];
RZZ(0.5*pi) q[18],q[10];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[26],q[12];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[32],q[14];
RZZ(0.5*pi) q[17],q[27];
RZZ(0.5*pi) q[37],q[19];
RZZ(0.5*pi) q[21],q[25];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[28],q[38];
U1q(0.350960357214425*pi,0.16258234565334995*pi) q[0];
U1q(0.53942981543986*pi,1.2145668856439809*pi) q[1];
U1q(0.475814349085179*pi,1.09805166666745*pi) q[2];
U1q(0.0671129059672122*pi,1.2111462136942004*pi) q[3];
U1q(0.460104857409945*pi,1.69630305719883*pi) q[4];
U1q(0.678882205031193*pi,1.29153207033319*pi) q[5];
U1q(0.422904016456739*pi,1.1356311927653402*pi) q[6];
U1q(0.496184424701934*pi,0.9832216853118592*pi) q[7];
U1q(0.184462558643782*pi,1.7964517821810304*pi) q[8];
U1q(0.432636566710256*pi,1.6519346323148296*pi) q[9];
U1q(0.299312754800996*pi,0.5910501808708601*pi) q[10];
U1q(0.78547344368643*pi,0.5520817530464797*pi) q[11];
U1q(0.493587616065517*pi,1.7347441993057195*pi) q[12];
U1q(0.357291804613787*pi,1.2465167141748896*pi) q[13];
U1q(0.279150577969848*pi,1.9474416834865496*pi) q[14];
U1q(0.170523676513637*pi,1.7169992174834103*pi) q[15];
U1q(0.320210901965476*pi,0.11570967822081002*pi) q[16];
U1q(0.479973440790331*pi,0.0007450887406701412*pi) q[17];
U1q(0.625405070773361*pi,0.36508257576980974*pi) q[18];
U1q(0.38953377482384*pi,1.35900305327874*pi) q[19];
U1q(0.423009593380798*pi,0.3396363777437905*pi) q[20];
U1q(0.304681699849085*pi,0.07717834663410983*pi) q[21];
U1q(0.458547791299337*pi,1.1720305746146504*pi) q[22];
U1q(0.224630691475019*pi,1.4783782855989598*pi) q[23];
U1q(0.583257278893331*pi,0.4318257773432004*pi) q[24];
U1q(0.300509906262804*pi,0.56306696502994*pi) q[25];
U1q(0.690899661764367*pi,0.19718779120467023*pi) q[26];
U1q(0.564133166828442*pi,1.7816920558359808*pi) q[27];
U1q(0.700723138103917*pi,1.7466379628039004*pi) q[28];
U1q(0.821533706065012*pi,1.7872698598262495*pi) q[29];
U1q(0.825176584670203*pi,1.9549797563654998*pi) q[30];
U1q(0.821947729029779*pi,1.4775583308278701*pi) q[31];
U1q(0.948321350206593*pi,1.6023428306228507*pi) q[32];
U1q(0.293326633647137*pi,1.6199166951049406*pi) q[33];
U1q(0.20173307561868*pi,0.0134007888331098*pi) q[34];
U1q(0.532848444814706*pi,1.9011507709031505*pi) q[35];
U1q(0.673956725064821*pi,1.6200239057447892*pi) q[36];
U1q(0.232279823166818*pi,1.0323136497087297*pi) q[37];
U1q(0.787528224128651*pi,0.1328017046368597*pi) q[38];
U1q(0.047159752077178*pi,0.5883572787441196*pi) q[39];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[39],q[1];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[32],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[29],q[10];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[16],q[12];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[20],q[33];
RZZ(0.5*pi) q[34],q[21];
RZZ(0.5*pi) q[26],q[22];
RZZ(0.5*pi) q[24],q[25];
RZZ(0.5*pi) q[28],q[27];
RZZ(0.5*pi) q[31],q[35];
RZZ(0.5*pi) q[36],q[38];
U1q(0.59005300278738*pi,1.2871434126495798*pi) q[0];
U1q(0.602423322555809*pi,0.9838091387003001*pi) q[1];
U1q(0.698475674612006*pi,0.009237848600180065*pi) q[2];
U1q(0.428128565691117*pi,1.5206702678867003*pi) q[3];
U1q(0.198802236290968*pi,1.47684943961365*pi) q[4];
U1q(0.877733414836215*pi,0.3832575694392997*pi) q[5];
U1q(0.400321340310302*pi,1.0387461891254004*pi) q[6];
U1q(0.33999221927338*pi,0.7817223033627005*pi) q[7];
U1q(0.204624656862554*pi,1.19465986956589*pi) q[8];
U1q(0.569769454204654*pi,0.6765380810591797*pi) q[9];
U1q(0.733154215175747*pi,0.31915280278883973*pi) q[10];
U1q(0.406903545890422*pi,0.021804637922269166*pi) q[11];
U1q(0.527660269531068*pi,0.1753502206560107*pi) q[12];
U1q(0.676748608399783*pi,1.6483646561012293*pi) q[13];
U1q(0.652301205050601*pi,0.5634251885106902*pi) q[14];
U1q(0.675431796585326*pi,0.09933216072179007*pi) q[15];
U1q(0.290201658645146*pi,1.8130869807389995*pi) q[16];
U1q(0.800116811198154*pi,0.9820260990715397*pi) q[17];
U1q(0.370874158774566*pi,0.5278636734219102*pi) q[18];
U1q(0.721206075371772*pi,0.9151836542052001*pi) q[19];
U1q(0.341196791458476*pi,0.9507053408690993*pi) q[20];
U1q(0.703321839395697*pi,1.4446847637426004*pi) q[21];
U1q(0.360978614193914*pi,0.3303377636282798*pi) q[22];
U1q(0.34041657180688*pi,1.4460018799870795*pi) q[23];
U1q(0.195262875074324*pi,1.1320864219162097*pi) q[24];
U1q(0.61031698088039*pi,0.6846304226304802*pi) q[25];
U1q(0.489155877198006*pi,1.54811841051961*pi) q[26];
U1q(0.711424557310894*pi,0.8104058059475996*pi) q[27];
U1q(0.576712550929494*pi,1.5901661532471998*pi) q[28];
U1q(0.695158359010583*pi,0.35511553324452017*pi) q[29];
U1q(0.320321136021936*pi,0.3969121548950998*pi) q[30];
U1q(0.43301006608237*pi,1.0022727539737009*pi) q[31];
U1q(0.207826706814355*pi,1.5304561838800002*pi) q[32];
U1q(0.527041481990146*pi,0.5350273232461404*pi) q[33];
U1q(0.425977409708461*pi,0.7119392014608401*pi) q[34];
U1q(0.64713446050936*pi,0.1107477717788008*pi) q[35];
U1q(0.80137203922011*pi,1.0539490853405997*pi) q[36];
U1q(0.483603022547699*pi,1.0155621786745392*pi) q[37];
U1q(0.337089429363408*pi,0.08414555433887028*pi) q[38];
U1q(0.417894143909869*pi,1.5322346967079898*pi) q[39];
RZZ(0.5*pi) q[26],q[0];
RZZ(0.5*pi) q[9],q[1];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[3],q[27];
RZZ(0.5*pi) q[4],q[37];
RZZ(0.5*pi) q[16],q[5];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[33];
RZZ(0.5*pi) q[8],q[21];
RZZ(0.5*pi) q[11],q[10];
RZZ(0.5*pi) q[36],q[12];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[32],q[17];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[23],q[38];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[30],q[34];
RZZ(0.5*pi) q[39],q[35];
U1q(0.287955770805544*pi,0.33687987545198084*pi) q[0];
U1q(0.396889261600719*pi,0.15680648682350018*pi) q[1];
U1q(0.811605063596355*pi,1.5563405907092704*pi) q[2];
U1q(0.342802557320546*pi,1.5472655037705003*pi) q[3];
U1q(0.708862314350867*pi,1.0141111765772006*pi) q[4];
U1q(0.323542731773479*pi,1.8679906236607007*pi) q[5];
U1q(0.683365208961152*pi,0.1144886860023604*pi) q[6];
U1q(0.433440683535776*pi,0.2366984722319998*pi) q[7];
U1q(0.412965858491314*pi,1.7185667585477002*pi) q[8];
U1q(0.760843530546528*pi,1.4854313669444004*pi) q[9];
U1q(0.515832048468667*pi,0.49192616683261026*pi) q[10];
U1q(0.184399905518353*pi,1.3115421102283005*pi) q[11];
U1q(0.284422443955993*pi,0.7405812801151992*pi) q[12];
U1q(0.390348376753942*pi,0.2841685960646494*pi) q[13];
U1q(0.46152997520879*pi,0.5665575105137606*pi) q[14];
U1q(0.801384553792422*pi,0.7238316702097993*pi) q[15];
U1q(0.475648397563756*pi,1.0073745853239*pi) q[16];
U1q(0.886608940681794*pi,0.4129785594052695*pi) q[17];
U1q(0.328012784678967*pi,1.9179992122152*pi) q[18];
U1q(0.251411684806948*pi,0.5997880299346399*pi) q[19];
U1q(0.425492004709638*pi,1.9440100073442004*pi) q[20];
U1q(0.422357738943971*pi,1.1963074437448995*pi) q[21];
U1q(0.677086995943338*pi,1.1812762438503803*pi) q[22];
U1q(0.369365741528123*pi,0.37819001775219974*pi) q[23];
U1q(0.396508582944539*pi,1.2378133077453999*pi) q[24];
U1q(0.25324074485502*pi,0.25897887588507995*pi) q[25];
U1q(0.939893350123918*pi,1.2654334087451993*pi) q[26];
U1q(0.255747064386983*pi,0.2988181182755998*pi) q[27];
U1q(0.426967784517062*pi,0.3412871193437006*pi) q[28];
U1q(0.551998194583386*pi,1.6828857848754009*pi) q[29];
U1q(0.925651875980164*pi,1.6844039609978*pi) q[30];
U1q(0.647534589467913*pi,0.9977966075210993*pi) q[31];
U1q(0.664333391981448*pi,0.8493378328040002*pi) q[32];
U1q(0.640834937249411*pi,0.015963835134499504*pi) q[33];
U1q(0.218237044123122*pi,0.5318832953155095*pi) q[34];
U1q(0.3848728516882*pi,1.5429669208552994*pi) q[35];
U1q(0.598334492003302*pi,1.2262806320315995*pi) q[36];
U1q(0.683592699783276*pi,1.7407221741768009*pi) q[37];
U1q(0.516784996550284*pi,1.9003626110313991*pi) q[38];
U1q(0.443006229781277*pi,1.55789781173765*pi) q[39];
RZZ(0.5*pi) q[0],q[28];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[30],q[3];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[5],q[35];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[7],q[11];
RZZ(0.5*pi) q[8],q[38];
RZZ(0.5*pi) q[26],q[10];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[29],q[23];
RZZ(0.5*pi) q[25],q[31];
RZZ(0.5*pi) q[39],q[27];
U1q(0.527705610358818*pi,1.9073815260027*pi) q[0];
U1q(0.234837684245971*pi,0.7150980978872994*pi) q[1];
U1q(0.439371099687314*pi,0.4032706086265101*pi) q[2];
U1q(0.42749792301588*pi,0.4845282773217008*pi) q[3];
U1q(0.438720516032248*pi,1.0080440937376007*pi) q[4];
U1q(0.246775791860013*pi,1.9384552633191987*pi) q[5];
U1q(0.322914818392338*pi,1.3291916473539196*pi) q[6];
U1q(0.690719002447034*pi,1.7184491246575*pi) q[7];
U1q(0.435508586416918*pi,1.2362652186266008*pi) q[8];
U1q(0.319243372648883*pi,1.6914157300343007*pi) q[9];
U1q(0.800675760182482*pi,1.5730457600982906*pi) q[10];
U1q(0.275930724241616*pi,1.0041950047277997*pi) q[11];
U1q(0.473876077922008*pi,0.5476307647622001*pi) q[12];
U1q(0.576183748953029*pi,0.7625524664883994*pi) q[13];
U1q(0.663003267924123*pi,1.2223708231641002*pi) q[14];
U1q(0.556272317547001*pi,1.0495185885843998*pi) q[15];
U1q(0.664907210289691*pi,0.3928276894696996*pi) q[16];
U1q(0.252230627198101*pi,1.9111157965503*pi) q[17];
U1q(0.444379201067929*pi,0.5112717432837997*pi) q[18];
U1q(0.264051100527358*pi,1.4122285761363997*pi) q[19];
U1q(0.31374127273082*pi,0.8381496363341991*pi) q[20];
U1q(0.344678077561337*pi,1.577709421830999*pi) q[21];
U1q(0.294909114270777*pi,1.6656233718015994*pi) q[22];
U1q(0.349284556502042*pi,0.8636135449166993*pi) q[23];
U1q(0.431362401086888*pi,0.9621726502544004*pi) q[24];
U1q(0.773192722957256*pi,1.0291136981420195*pi) q[25];
U1q(0.626507365870907*pi,1.0706650510801001*pi) q[26];
U1q(0.14616051072398*pi,1.0745200259947012*pi) q[27];
U1q(0.481265979507934*pi,1.3794096850124014*pi) q[28];
U1q(0.546109333460938*pi,0.1931906656413993*pi) q[29];
U1q(0.831344664048839*pi,1.0248970281980014*pi) q[30];
U1q(0.539441374070976*pi,1.9781148747245005*pi) q[31];
U1q(0.519425423311033*pi,1.9519047983153008*pi) q[32];
U1q(0.409347809457449*pi,0.14985561564770045*pi) q[33];
U1q(0.24297376457647*pi,0.20754891804959996*pi) q[34];
U1q(0.68793675981449*pi,1.3127938410779993*pi) q[35];
U1q(0.317735858569262*pi,1.4667858059778993*pi) q[36];
U1q(0.531721121461977*pi,1.7702284244998001*pi) q[37];
U1q(0.499238108741128*pi,0.8701625128231001*pi) q[38];
U1q(0.92974268238323*pi,1.6936573766615997*pi) q[39];
RZZ(0.5*pi) q[14],q[0];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[18],q[4];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[7],q[6];
RZZ(0.5*pi) q[8],q[12];
RZZ(0.5*pi) q[29],q[9];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[16],q[13];
RZZ(0.5*pi) q[17],q[39];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[23],q[20];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[22],q[28];
RZZ(0.5*pi) q[30],q[24];
RZZ(0.5*pi) q[25],q[38];
RZZ(0.5*pi) q[36],q[27];
RZZ(0.5*pi) q[37],q[31];
RZZ(0.5*pi) q[32],q[33];
U1q(0.596577776091689*pi,1.8702285811856*pi) q[0];
U1q(0.71656703957043*pi,1.168426050450499*pi) q[1];
U1q(0.0425487592979665*pi,0.5637251480557008*pi) q[2];
U1q(0.716710337452568*pi,1.0901384969744008*pi) q[3];
U1q(0.79889003584272*pi,0.1948555108086012*pi) q[4];
U1q(0.772261568527513*pi,0.8899435754574014*pi) q[5];
U1q(0.694326014996532*pi,0.32943872913140027*pi) q[6];
U1q(0.735616435568278*pi,0.7373162141341005*pi) q[7];
U1q(0.721937128777114*pi,0.06961071431610044*pi) q[8];
U1q(0.552964948053605*pi,1.2971785994871006*pi) q[9];
U1q(0.318309624486354*pi,1.9746621578903003*pi) q[10];
U1q(0.245735973317915*pi,0.4143594377731006*pi) q[11];
U1q(0.192663427527531*pi,1.5376883098011014*pi) q[12];
U1q(0.723302944265099*pi,1.6257303074649005*pi) q[13];
U1q(0.0382646487522024*pi,0.8165727841379997*pi) q[14];
U1q(0.846590435186206*pi,1.2929783569035997*pi) q[15];
U1q(0.56098033511947*pi,1.9238723320335005*pi) q[16];
U1q(0.714259647317633*pi,1.7708019960005004*pi) q[17];
U1q(0.563589727970716*pi,0.5154491738095999*pi) q[18];
U1q(0.287101387735684*pi,1.9506814897757998*pi) q[19];
U1q(0.441437676620964*pi,1.331287568935501*pi) q[20];
U1q(0.300864817679405*pi,0.4713017992101989*pi) q[21];
U1q(0.585519184460582*pi,1.5795340645573006*pi) q[22];
U1q(0.985621179665091*pi,1.3738860495293004*pi) q[23];
U1q(0.301122779327592*pi,0.10174718827549967*pi) q[24];
U1q(0.53003174059551*pi,1.8409983107654995*pi) q[25];
U1q(0.858324221143678*pi,0.07848019788599991*pi) q[26];
U1q(0.359519979763811*pi,0.4456825918294989*pi) q[27];
U1q(0.687241242342208*pi,1.1028919398364003*pi) q[28];
U1q(0.522289182806848*pi,1.3431181500166005*pi) q[29];
U1q(0.513814225979118*pi,1.8089068904343009*pi) q[30];
U1q(0.572460263027626*pi,1.3389340651439987*pi) q[31];
U1q(0.285306819378541*pi,0.9012677185904998*pi) q[32];
U1q(0.607035005697297*pi,1.4659190783828002*pi) q[33];
U1q(0.333281188027353*pi,1.2879956974402003*pi) q[34];
U1q(0.414346671613251*pi,0.9021371308595008*pi) q[35];
U1q(0.761681968175584*pi,1.8224133947754986*pi) q[36];
U1q(0.49158199990253*pi,1.4965592039967994*pi) q[37];
U1q(0.170254237309334*pi,1.412095770062301*pi) q[38];
U1q(0.601624416973499*pi,0.9548588068133999*pi) q[39];
RZZ(0.5*pi) q[16],q[0];
RZZ(0.5*pi) q[12],q[1];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[28],q[3];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[9],q[13];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[14],q[35];
RZZ(0.5*pi) q[30],q[17];
RZZ(0.5*pi) q[26],q[18];
RZZ(0.5*pi) q[32],q[19];
RZZ(0.5*pi) q[22],q[33];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[24],q[31];
RZZ(0.5*pi) q[36],q[25];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[34],q[37];
U1q(0.758060549272881*pi,0.4030863702432015*pi) q[0];
U1q(0.577821725148086*pi,0.1712538706260993*pi) q[1];
U1q(0.583629501378455*pi,1.9443029764914002*pi) q[2];
U1q(0.649691113183676*pi,1.7898150240994006*pi) q[3];
U1q(0.207398794045419*pi,0.6957526293663996*pi) q[4];
U1q(0.632508418240565*pi,0.821537393254701*pi) q[5];
U1q(0.786798578815213*pi,0.4847847840308006*pi) q[6];
U1q(0.693116059004107*pi,0.44056291634570144*pi) q[7];
U1q(0.645114478607619*pi,0.7463215491289006*pi) q[8];
U1q(0.853114965533965*pi,1.4454373491267987*pi) q[9];
U1q(0.505425052080061*pi,1.4416976633605003*pi) q[10];
U1q(0.595088238586315*pi,0.8554757767880012*pi) q[11];
U1q(0.744862823876067*pi,0.02147458763329979*pi) q[12];
U1q(0.784047189021052*pi,1.0699680833161*pi) q[13];
U1q(0.880450909137428*pi,1.6320483629402993*pi) q[14];
U1q(0.431993445157328*pi,0.9329750907973988*pi) q[15];
U1q(0.34758651345419*pi,0.5652473586782989*pi) q[16];
U1q(0.250239708698206*pi,1.5189507344998*pi) q[17];
U1q(0.654075489460009*pi,0.8727273407813989*pi) q[18];
U1q(0.417858773166908*pi,1.0785275348982992*pi) q[19];
U1q(0.575796800000364*pi,1.0221738222945014*pi) q[20];
U1q(0.567357783659513*pi,1.6799827245197*pi) q[21];
U1q(0.720329960563811*pi,0.9777970886158993*pi) q[22];
U1q(0.113892139795038*pi,1.6157240993729012*pi) q[23];
U1q(0.499287910177705*pi,0.10047895932260076*pi) q[24];
U1q(0.274113626736141*pi,0.010719295457599642*pi) q[25];
U1q(0.251336744279733*pi,1.1132145451895994*pi) q[26];
U1q(0.733372112763374*pi,0.15634720457650175*pi) q[27];
U1q(0.474094537282333*pi,1.8775948473328015*pi) q[28];
U1q(0.840412135754575*pi,0.4810348641240001*pi) q[29];
U1q(0.567471496231594*pi,1.4947383770211005*pi) q[30];
U1q(0.738688606675631*pi,0.5084886372800987*pi) q[31];
U1q(0.546212730605076*pi,1.271119761962801*pi) q[32];
U1q(0.310678348976708*pi,1.9744840843478002*pi) q[33];
U1q(0.742431651976364*pi,0.14476825875929933*pi) q[34];
U1q(0.583193416648434*pi,0.2869858374183991*pi) q[35];
U1q(0.826755441436417*pi,1.0537060153714997*pi) q[36];
U1q(0.874315571112304*pi,1.7338330092802003*pi) q[37];
U1q(0.279584224575831*pi,1.8034674017537*pi) q[38];
U1q(0.339487831514819*pi,1.6389361057805*pi) q[39];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[33];
RZZ(0.5*pi) q[4],q[28];
RZZ(0.5*pi) q[6],q[5];
RZZ(0.5*pi) q[14],q[8];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[20],q[10];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[26],q[17];
RZZ(0.5*pi) q[24],q[19];
RZZ(0.5*pi) q[39],q[21];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[34],q[23];
RZZ(0.5*pi) q[29],q[32];
RZZ(0.5*pi) q[36],q[35];
U1q(0.68057942213721*pi,1.8041689664166007*pi) q[0];
U1q(0.868902280112407*pi,1.9694195249297017*pi) q[1];
U1q(0.285518421175135*pi,1.5260346367314988*pi) q[2];
U1q(0.461478358552844*pi,0.16778823936009957*pi) q[3];
U1q(0.724876841165959*pi,0.8341280447371986*pi) q[4];
U1q(0.542907523792518*pi,1.706481139574901*pi) q[5];
U1q(0.43411667665432*pi,1.9933533143749997*pi) q[6];
U1q(0.877087833483103*pi,0.5759566313881983*pi) q[7];
U1q(0.392809675674949*pi,0.09584315448239877*pi) q[8];
U1q(0.439619833882318*pi,1.8101375391956012*pi) q[9];
U1q(0.415717378347106*pi,0.6749477299016995*pi) q[10];
U1q(0.389700143722378*pi,0.8585234110188011*pi) q[11];
U1q(0.582085743752876*pi,1.6303854861562002*pi) q[12];
U1q(0.919498742557044*pi,0.6181502903980007*pi) q[13];
U1q(0.217674967296621*pi,0.6225828942355989*pi) q[14];
U1q(0.843167974410905*pi,1.4859326281758989*pi) q[15];
U1q(0.931689017394729*pi,0.4213385135461998*pi) q[16];
U1q(0.187409355309953*pi,0.5027992782825983*pi) q[17];
U1q(0.376550108304384*pi,0.5381508253963005*pi) q[18];
U1q(0.801902007044097*pi,1.8248157781544982*pi) q[19];
U1q(0.485109676894629*pi,0.6995173142429998*pi) q[20];
U1q(0.557561185864675*pi,1.8044663953874007*pi) q[21];
U1q(0.0891069259583392*pi,1.3788719711109998*pi) q[22];
U1q(0.437581638735259*pi,0.9124239527782017*pi) q[23];
U1q(0.562251454642787*pi,0.1535848130295001*pi) q[24];
U1q(0.879094965928558*pi,1.6185991834529005*pi) q[25];
U1q(0.51522066771732*pi,1.7700675128666*pi) q[26];
U1q(0.675834392441199*pi,1.695591932885499*pi) q[27];
U1q(0.176547507226574*pi,1.0275790093709993*pi) q[28];
U1q(0.347566024216946*pi,1.8460885901705986*pi) q[29];
U1q(0.643159286310213*pi,1.0033458055396984*pi) q[30];
U1q(0.511818004756006*pi,0.5938143937256015*pi) q[31];
U1q(0.329957700955903*pi,1.0945995728386002*pi) q[32];
U1q(0.565926349519792*pi,0.9695770708797014*pi) q[33];
U1q(0.715406683014391*pi,1.0991279448324*pi) q[34];
U1q(0.280538264425988*pi,1.0497700649465997*pi) q[35];
U1q(0.731093217838255*pi,0.22455642808129994*pi) q[36];
U1q(0.172973108606004*pi,1.174176491223001*pi) q[37];
U1q(0.441096802006568*pi,1.1493185542211002*pi) q[38];
U1q(0.54570736358958*pi,1.4621925113776992*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[2],q[38];
RZZ(0.5*pi) q[37],q[3];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[13],q[5];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[9],q[39];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[15],q[16];
RZZ(0.5*pi) q[23],q[17];
RZZ(0.5*pi) q[29],q[18];
RZZ(0.5*pi) q[20],q[35];
RZZ(0.5*pi) q[24],q[21];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[28],q[25];
RZZ(0.5*pi) q[33],q[31];
U1q(0.737802969338577*pi,1.9473543248687015*pi) q[0];
U1q(0.764656003263939*pi,0.7510909391140999*pi) q[1];
U1q(0.465028902493141*pi,0.5102443275995014*pi) q[2];
U1q(0.429146600802269*pi,0.5500166337454999*pi) q[3];
U1q(0.681563579511893*pi,1.9693788795373983*pi) q[4];
U1q(0.837951636298385*pi,0.2281862911383996*pi) q[5];
U1q(0.859696426681609*pi,0.7631519380953016*pi) q[6];
U1q(0.129167643149502*pi,0.6775868146873982*pi) q[7];
U1q(0.767719114459805*pi,0.33847726094539965*pi) q[8];
U1q(0.820084354925308*pi,1.5887049256771988*pi) q[9];
U1q(0.330337969102693*pi,1.9818685399712983*pi) q[10];
U1q(0.356959858195478*pi,1.544294053825599*pi) q[11];
U1q(0.343639385867721*pi,0.8554768126309007*pi) q[12];
U1q(0.334857626083143*pi,1.885987342904901*pi) q[13];
U1q(0.926533985762231*pi,1.5975413000397012*pi) q[14];
U1q(0.318645158100716*pi,0.6993009042564005*pi) q[15];
U1q(0.666066033085469*pi,1.696000228326401*pi) q[16];
U1q(0.441183388563822*pi,0.5674593797896001*pi) q[17];
U1q(0.271995702330767*pi,1.143438670618199*pi) q[18];
U1q(0.323526977579818*pi,0.9936435719764987*pi) q[19];
U1q(0.606792566080712*pi,1.5035545213067003*pi) q[20];
U1q(0.556645150682799*pi,0.9319691010236006*pi) q[21];
U1q(0.712868622989503*pi,1.1429951771684017*pi) q[22];
U1q(0.438270390044386*pi,0.6378831539560004*pi) q[23];
U1q(0.250727053149399*pi,1.9742101984332017*pi) q[24];
U1q(0.617100896393522*pi,1.6277417003674*pi) q[25];
U1q(0.6678393336705*pi,1.1931606743601009*pi) q[26];
U1q(0.640707796684125*pi,0.10115624940069878*pi) q[27];
U1q(0.487804920517961*pi,1.2975138339010996*pi) q[28];
U1q(0.633120672162483*pi,1.4817786667239012*pi) q[29];
U1q(0.614082107939381*pi,1.5809716258761988*pi) q[30];
U1q(0.0864989846961296*pi,1.3679074927491008*pi) q[31];
U1q(0.742650359674858*pi,0.9319187032587983*pi) q[32];
U1q(0.72928055195711*pi,0.46944821934350145*pi) q[33];
U1q(0.230227091995127*pi,0.36610648399570067*pi) q[34];
U1q(0.212226035922637*pi,0.8868188066496003*pi) q[35];
U1q(0.615041805383147*pi,1.0750012218822*pi) q[36];
U1q(0.02241491933372*pi,1.0716568485975984*pi) q[37];
U1q(0.529724483681371*pi,0.48516308659580076*pi) q[38];
U1q(0.155138278243226*pi,0.2138669343156998*pi) q[39];
rz(2.9234683918321984*pi) q[0];
rz(0.8648384974704015*pi) q[1];
rz(1.9803712091589993*pi) q[2];
rz(1.6945581788266004*pi) q[3];
rz(1.6542623879923006*pi) q[4];
rz(2.2786868851394004*pi) q[5];
rz(1.4510906443804998*pi) q[6];
rz(2.173000942784899*pi) q[7];
rz(1.058042982424901*pi) q[8];
rz(1.6422876760657985*pi) q[9];
rz(1.7762371970252993*pi) q[10];
rz(1.2536448106441007*pi) q[11];
rz(1.9979741544328*pi) q[12];
rz(1.9557615858813016*pi) q[13];
rz(3.7175139686415015*pi) q[14];
rz(2.263737390558699*pi) q[15];
rz(0.20120692697210174*pi) q[16];
rz(1.6849666995618016*pi) q[17];
rz(3.2451953117627994*pi) q[18];
rz(1.450931464618499*pi) q[19];
rz(1.1410438088981003*pi) q[20];
rz(1.1738796020663003*pi) q[21];
rz(3.7400223562273*pi) q[22];
rz(0.46201923478989926*pi) q[23];
rz(2.0758614038386014*pi) q[24];
rz(2.502059879689501*pi) q[25];
rz(2.778527691399299*pi) q[26];
rz(0.8368689242461009*pi) q[27];
rz(3.006267016158901*pi) q[28];
rz(0.9498576839807988*pi) q[29];
rz(1.2737516650954*pi) q[30];
rz(0.3164673439777985*pi) q[31];
rz(2.3974666800902007*pi) q[32];
rz(3.8351690991410017*pi) q[33];
rz(1.1197781441451014*pi) q[34];
rz(2.4726077954892*pi) q[35];
rz(1.1318113860919006*pi) q[36];
rz(1.7431890665804985*pi) q[37];
rz(1.0766642347455004*pi) q[38];
rz(0.2840159308163983*pi) q[39];
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