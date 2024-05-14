OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.0768038059050691*pi,0.9305332381934599*pi) q[0];
U1q(0.722318170594615*pi,1.24608898004308*pi) q[1];
U1q(0.653312531832154*pi,0.422251510499775*pi) q[2];
U1q(0.440262118037798*pi,1.3504106250583812*pi) q[3];
U1q(0.537110061502313*pi,0.694778698870841*pi) q[4];
U1q(0.394788816206737*pi,1.860212581409961*pi) q[5];
U1q(0.634767378004688*pi,1.493679738238812*pi) q[6];
U1q(0.141108122828852*pi,0.534669224540355*pi) q[7];
U1q(0.903734955187421*pi,1.60836126552555*pi) q[8];
U1q(0.430137174001284*pi,0.365017908357198*pi) q[9];
U1q(0.516525732991594*pi,0.805939683297313*pi) q[10];
U1q(0.901522612069899*pi,0.232236170267092*pi) q[11];
U1q(0.158024885298942*pi,0.54157006050266*pi) q[12];
U1q(0.474700759551451*pi,1.049676694272201*pi) q[13];
U1q(0.539291121803825*pi,1.23879099394434*pi) q[14];
U1q(0.375936113446171*pi,0.0459697405594403*pi) q[15];
U1q(0.829287328177616*pi,1.827882338345133*pi) q[16];
U1q(0.760094140860553*pi,0.0881165485338836*pi) q[17];
U1q(0.840394148467611*pi,0.214540790730058*pi) q[18];
U1q(0.727623070720859*pi,1.475821237543785*pi) q[19];
U1q(0.325781715816912*pi,0.0747147657841312*pi) q[20];
U1q(0.136277745670769*pi,1.9198894481455038*pi) q[21];
U1q(0.184481272687103*pi,1.864664108946157*pi) q[22];
U1q(0.411623301461751*pi,1.565323266233041*pi) q[23];
U1q(0.55441758115359*pi,1.559525802365299*pi) q[24];
U1q(0.680586059761649*pi,0.872723945587696*pi) q[25];
U1q(0.505505184149721*pi,0.0422118469329233*pi) q[26];
U1q(0.465236276688716*pi,1.5222309072238591*pi) q[27];
U1q(0.674597493534309*pi,0.867440150415956*pi) q[28];
U1q(0.581996971095421*pi,1.739535267770905*pi) q[29];
U1q(0.362413014851458*pi,1.380043333147618*pi) q[30];
U1q(0.20821507346776*pi,0.0126863603132221*pi) q[31];
U1q(0.630429467363411*pi,1.16974775225012*pi) q[32];
U1q(0.829524871137585*pi,1.375682650449749*pi) q[33];
U1q(0.457894478078008*pi,0.518123811352566*pi) q[34];
U1q(0.191735306381067*pi,1.210032112331273*pi) q[35];
U1q(0.718428464058954*pi,0.96079381719707*pi) q[36];
U1q(0.368028783313869*pi,0.4466762098642101*pi) q[37];
U1q(0.590559364157603*pi,0.498880641469322*pi) q[38];
U1q(0.375251547803387*pi,1.04730124932551*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[25],q[1];
RZZ(0.5*pi) q[13],q[2];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[30],q[6];
RZZ(0.5*pi) q[39],q[7];
RZZ(0.5*pi) q[8],q[36];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[10],q[35];
RZZ(0.5*pi) q[29],q[11];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[15],q[38];
RZZ(0.5*pi) q[32],q[17];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[19],q[23];
RZZ(0.5*pi) q[27],q[21];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[24],q[26];
U1q(0.567830491213801*pi,1.78314131234281*pi) q[0];
U1q(0.468546359081579*pi,0.7238792075515199*pi) q[1];
U1q(0.483398662974049*pi,0.08872945231731011*pi) q[2];
U1q(0.251943321484826*pi,1.05476920640969*pi) q[3];
U1q(0.661778625438252*pi,1.158759264595917*pi) q[4];
U1q(0.51452266718606*pi,1.59576270180898*pi) q[5];
U1q(0.195578084821081*pi,1.4665476785545*pi) q[6];
U1q(0.351873903194111*pi,0.0814827238170499*pi) q[7];
U1q(0.471306560567861*pi,1.878608749838776*pi) q[8];
U1q(0.332776169335004*pi,1.3447254949141199*pi) q[9];
U1q(0.57051830369621*pi,1.514118954384175*pi) q[10];
U1q(0.855951519005744*pi,0.83139647771137*pi) q[11];
U1q(0.247169404771359*pi,1.2481839208010501*pi) q[12];
U1q(0.404716456167035*pi,1.77830765362175*pi) q[13];
U1q(0.425479771026583*pi,0.9553516879425299*pi) q[14];
U1q(0.388517521736302*pi,0.43224995295576996*pi) q[15];
U1q(0.397790595217513*pi,1.4628453626784301*pi) q[16];
U1q(0.0893133927662545*pi,1.334315382400037*pi) q[17];
U1q(0.597695301875754*pi,0.34072676869685004*pi) q[18];
U1q(0.130252734925892*pi,0.18337794973502985*pi) q[19];
U1q(0.333990708601751*pi,1.0244297238767102*pi) q[20];
U1q(0.201149947714466*pi,0.72581291366747*pi) q[21];
U1q(0.567712683177228*pi,0.5308952617756799*pi) q[22];
U1q(0.46552345403707*pi,1.9095844121273902*pi) q[23];
U1q(0.743237204702218*pi,1.79680475710221*pi) q[24];
U1q(0.602262181167191*pi,1.0316507272838*pi) q[25];
U1q(0.564929313064099*pi,1.29233687370649*pi) q[26];
U1q(0.50411898348542*pi,1.33851060540204*pi) q[27];
U1q(0.267857017330474*pi,0.4067405327470901*pi) q[28];
U1q(0.510942395228997*pi,1.22837951561058*pi) q[29];
U1q(0.745284265761598*pi,1.04904617849017*pi) q[30];
U1q(0.944679420072933*pi,1.33678282166251*pi) q[31];
U1q(0.426236139974479*pi,0.4399942488094599*pi) q[32];
U1q(0.827514615059839*pi,0.9216770681542101*pi) q[33];
U1q(0.459804023020755*pi,1.34321132757142*pi) q[34];
U1q(0.591692788494838*pi,0.79239898462427*pi) q[35];
U1q(0.390671704109857*pi,1.7277006165058797*pi) q[36];
U1q(0.711898283942797*pi,1.6253233698543799*pi) q[37];
U1q(0.649503789224282*pi,1.4413171390761699*pi) q[38];
U1q(0.889479901610409*pi,0.5100880766026801*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[32],q[5];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[35],q[11];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[24],q[13];
RZZ(0.5*pi) q[37],q[14];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[16],q[26];
RZZ(0.5*pi) q[38],q[17];
RZZ(0.5*pi) q[18],q[22];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[33],q[28];
RZZ(0.5*pi) q[29],q[36];
U1q(0.53383331327582*pi,0.9090187346265397*pi) q[0];
U1q(0.356936371558013*pi,0.09909234204118*pi) q[1];
U1q(0.437467221735415*pi,0.89520635824507*pi) q[2];
U1q(0.78837933405644*pi,1.5567336535842804*pi) q[3];
U1q(0.275576867742153*pi,0.6492191291718199*pi) q[4];
U1q(0.374955136358845*pi,0.38145635576164016*pi) q[5];
U1q(0.353806407993877*pi,0.42323132526011964*pi) q[6];
U1q(0.368261325565621*pi,1.32460723502993*pi) q[7];
U1q(0.58271665995656*pi,0.51668428560698*pi) q[8];
U1q(0.382680797730498*pi,1.0908318266105397*pi) q[9];
U1q(0.890016378628254*pi,1.0730038159550102*pi) q[10];
U1q(0.6251981646226*pi,0.11492270806868987*pi) q[11];
U1q(0.6964891686957*pi,0.4961047439270998*pi) q[12];
U1q(0.572302623230266*pi,0.18148167469986998*pi) q[13];
U1q(0.871911612695767*pi,0.42321392017578985*pi) q[14];
U1q(0.397367475683466*pi,0.48708241616829984*pi) q[15];
U1q(0.540746794598932*pi,0.5210579685313301*pi) q[16];
U1q(0.651731778651981*pi,0.2377499111967598*pi) q[17];
U1q(0.428093991104333*pi,1.26788034321916*pi) q[18];
U1q(0.571690718302509*pi,0.6720639156727701*pi) q[19];
U1q(0.757873214693935*pi,1.3089882364442902*pi) q[20];
U1q(0.431768836493347*pi,1.9044462350583*pi) q[21];
U1q(0.170670489921311*pi,0.8134004546693698*pi) q[22];
U1q(0.52276570994159*pi,1.5773533878610104*pi) q[23];
U1q(0.485702364334735*pi,0.9909130120422498*pi) q[24];
U1q(0.932146215982113*pi,0.880811318646294*pi) q[25];
U1q(0.52507614212575*pi,0.25914825592810997*pi) q[26];
U1q(0.529956406521114*pi,1.9078529682311203*pi) q[27];
U1q(0.152732076563437*pi,0.12505821915617998*pi) q[28];
U1q(0.46368211765627*pi,0.28858372858065984*pi) q[29];
U1q(0.272682426653855*pi,0.76367515842204*pi) q[30];
U1q(0.499706622751037*pi,1.57025800265923*pi) q[31];
U1q(0.676872190419743*pi,0.8696906772314801*pi) q[32];
U1q(0.287307997396022*pi,1.24330007247912*pi) q[33];
U1q(0.69940567696526*pi,0.6348142526378302*pi) q[34];
U1q(0.496119490303055*pi,0.4934255690239704*pi) q[35];
U1q(0.542202272364703*pi,1.5622345425706197*pi) q[36];
U1q(0.529553552774838*pi,1.1260891789112701*pi) q[37];
U1q(0.496320325167272*pi,1.16355679571899*pi) q[38];
U1q(0.308905593703859*pi,1.1930628234402998*pi) q[39];
RZZ(0.5*pi) q[33],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[37],q[2];
RZZ(0.5*pi) q[30],q[4];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[16],q[7];
RZZ(0.5*pi) q[10],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[17],q[12];
RZZ(0.5*pi) q[13],q[14];
RZZ(0.5*pi) q[18],q[23];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[22],q[20];
RZZ(0.5*pi) q[24],q[21];
RZZ(0.5*pi) q[25],q[36];
RZZ(0.5*pi) q[26],q[31];
RZZ(0.5*pi) q[38],q[27];
RZZ(0.5*pi) q[34],q[28];
RZZ(0.5*pi) q[39],q[32];
U1q(0.237881380237431*pi,1.8463547648526006*pi) q[0];
U1q(0.703603329335589*pi,1.21676896598348*pi) q[1];
U1q(0.316539178327575*pi,0.40420240703565025*pi) q[2];
U1q(0.56785118322208*pi,0.26230604756746967*pi) q[3];
U1q(0.536588939577932*pi,1.8487650399600302*pi) q[4];
U1q(0.788654774060188*pi,1.5026165371263502*pi) q[5];
U1q(0.704250774736605*pi,1.9181151799277494*pi) q[6];
U1q(0.684495811315647*pi,1.59332121866733*pi) q[7];
U1q(0.664156099935307*pi,1.9315570265890996*pi) q[8];
U1q(0.640352581006852*pi,0.09891774421263033*pi) q[9];
U1q(0.781239497840973*pi,1.1952549949634497*pi) q[10];
U1q(0.363043511096478*pi,1.2143753579192698*pi) q[11];
U1q(0.628709233818082*pi,1.6498932032879097*pi) q[12];
U1q(0.11262407736128*pi,1.85381151920852*pi) q[13];
U1q(0.0881483313395786*pi,1.29075364783239*pi) q[14];
U1q(0.895446377946746*pi,1.9095479222919698*pi) q[15];
U1q(0.822701139536368*pi,0.5218345742009696*pi) q[16];
U1q(0.342188241320096*pi,1.7769305977088798*pi) q[17];
U1q(0.505941450569374*pi,0.3354027016097101*pi) q[18];
U1q(0.927955143460411*pi,0.08968491587097027*pi) q[19];
U1q(0.494651983544495*pi,1.7745842237425506*pi) q[20];
U1q(0.395095260856836*pi,1.70310631629855*pi) q[21];
U1q(0.379597057853591*pi,0.3216480987010897*pi) q[22];
U1q(0.802184431285352*pi,1.78863428325805*pi) q[23];
U1q(0.694864279021214*pi,0.3570636511280396*pi) q[24];
U1q(0.747686125612676*pi,1.9927663841948633*pi) q[25];
U1q(0.749426844463351*pi,0.71094814649532*pi) q[26];
U1q(0.878853949729212*pi,1.3303043166296202*pi) q[27];
U1q(0.397046098521042*pi,1.9554877690702597*pi) q[28];
U1q(0.373844349723428*pi,0.46304779027358034*pi) q[29];
U1q(0.228368052707718*pi,0.5693058232186496*pi) q[30];
U1q(0.594468220355444*pi,1.0429379677278003*pi) q[31];
U1q(0.244201188063605*pi,1.1078260070813402*pi) q[32];
U1q(0.471846887346386*pi,0.39421932345219*pi) q[33];
U1q(0.340442690418758*pi,0.7501695693255499*pi) q[34];
U1q(0.391654055498087*pi,1.7367946646064096*pi) q[35];
U1q(0.951135350216457*pi,0.9452460484465597*pi) q[36];
U1q(0.596526433189091*pi,1.7307824397945009*pi) q[37];
U1q(0.61552967665079*pi,0.9800037348078297*pi) q[38];
U1q(0.394171002275449*pi,0.5969146745016101*pi) q[39];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[5],q[1];
RZZ(0.5*pi) q[38],q[2];
RZZ(0.5*pi) q[26],q[3];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[33],q[8];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[30],q[11];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[18],q[39];
RZZ(0.5*pi) q[32],q[19];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[22],q[27];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[34],q[25];
U1q(0.431489993464546*pi,1.3251481253358008*pi) q[0];
U1q(0.154895498526475*pi,0.3160671579071508*pi) q[1];
U1q(0.646988136197167*pi,1.7974155937601903*pi) q[2];
U1q(0.187716105775625*pi,1.9644690195992496*pi) q[3];
U1q(0.522358996090166*pi,1.2230589337589102*pi) q[4];
U1q(0.179708156273425*pi,0.8232230535090697*pi) q[5];
U1q(0.672630496405828*pi,0.9409318694328999*pi) q[6];
U1q(0.492401518634231*pi,0.45111037316442015*pi) q[7];
U1q(0.862515352757307*pi,1.72037408571898*pi) q[8];
U1q(0.544903311502954*pi,0.8008142773640596*pi) q[9];
U1q(0.576779056310514*pi,1.2581427834481298*pi) q[10];
U1q(0.572232013294638*pi,0.53410916287371*pi) q[11];
U1q(0.40389483053313*pi,0.04149379841857037*pi) q[12];
U1q(0.614984540707671*pi,1.23022277077985*pi) q[13];
U1q(0.725368437080814*pi,1.0203768644524303*pi) q[14];
U1q(0.275813511785188*pi,0.8905397529366796*pi) q[15];
U1q(0.630065841458217*pi,1.4632378917951803*pi) q[16];
U1q(0.300798442086509*pi,1.6426710716658892*pi) q[17];
U1q(0.295664047906249*pi,0.6142419544815407*pi) q[18];
U1q(0.108203740908913*pi,1.3712679871442006*pi) q[19];
U1q(0.400574936376868*pi,1.0068253129854607*pi) q[20];
U1q(0.596638467055393*pi,0.5045384276849596*pi) q[21];
U1q(0.48561024492301*pi,0.6173022130861092*pi) q[22];
U1q(0.711815046362189*pi,1.2577311178224502*pi) q[23];
U1q(0.365195638235069*pi,1.6773420515948008*pi) q[24];
U1q(0.752207835641319*pi,0.53791870484283*pi) q[25];
U1q(0.136657573052868*pi,1.7910551597929096*pi) q[26];
U1q(0.822571771440568*pi,0.53567622474998*pi) q[27];
U1q(0.35471972993149*pi,0.34770842449399986*pi) q[28];
U1q(0.558946987204442*pi,0.2894334760960007*pi) q[29];
U1q(0.637761822857964*pi,0.7619267755895809*pi) q[30];
U1q(0.363043335865902*pi,1.0063406520073404*pi) q[31];
U1q(0.587461059377409*pi,1.8223838687835396*pi) q[32];
U1q(0.238918356276169*pi,1.1974603558359007*pi) q[33];
U1q(0.754877791563417*pi,1.1595880611696892*pi) q[34];
U1q(0.674498137332729*pi,0.9886874728786097*pi) q[35];
U1q(0.645584788928749*pi,0.7510076219107003*pi) q[36];
U1q(0.437959185642711*pi,0.6696743668774001*pi) q[37];
U1q(0.39118352037645*pi,1.1137654149632699*pi) q[38];
U1q(0.765550621144639*pi,0.8770741341034398*pi) q[39];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[38],q[1];
RZZ(0.5*pi) q[19],q[2];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[4],q[31];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[23];
RZZ(0.5*pi) q[25],q[8];
RZZ(0.5*pi) q[24],q[9];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[35],q[13];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[17];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[28],q[21];
RZZ(0.5*pi) q[22],q[37];
RZZ(0.5*pi) q[34],q[29];
RZZ(0.5*pi) q[32],q[30];
U1q(0.41631787361103*pi,1.1449169292427008*pi) q[0];
U1q(0.362405973234054*pi,0.05797099029950026*pi) q[1];
U1q(0.521584628500043*pi,0.10972581932123049*pi) q[2];
U1q(0.846576991096256*pi,0.4875288009437604*pi) q[3];
U1q(0.271140870974206*pi,0.11108031661512996*pi) q[4];
U1q(0.30591439226077*pi,0.24740396092959926*pi) q[5];
U1q(0.264227039082433*pi,0.9253915951398*pi) q[6];
U1q(0.442459263608333*pi,1.9079698275845*pi) q[7];
U1q(0.267147283748032*pi,1.77412815085603*pi) q[8];
U1q(0.399817149603484*pi,1.3035949010041996*pi) q[9];
U1q(0.853655156982509*pi,0.10380049488951038*pi) q[10];
U1q(0.522481551980278*pi,0.9427289776154009*pi) q[11];
U1q(0.487579157789772*pi,1.7113732678250209*pi) q[12];
U1q(0.333286119746273*pi,0.9865145180588009*pi) q[13];
U1q(0.674520694843131*pi,1.9383697233584005*pi) q[14];
U1q(0.77585266071587*pi,0.8031732293897402*pi) q[15];
U1q(0.734477625642625*pi,1.0297151976256904*pi) q[16];
U1q(0.562441551853992*pi,0.8788992203568995*pi) q[17];
U1q(0.6609651311616*pi,0.09021200611990032*pi) q[18];
U1q(0.517943507199965*pi,1.1333569569851996*pi) q[19];
U1q(0.509547608259143*pi,1.7019456025191992*pi) q[20];
U1q(0.701478437220168*pi,0.36313939005579954*pi) q[21];
U1q(0.804762835265958*pi,1.5841547335663009*pi) q[22];
U1q(0.274921626408929*pi,0.6151574038511605*pi) q[23];
U1q(0.546025577856299*pi,1.0166305307238002*pi) q[24];
U1q(0.445151047482838*pi,1.7257811000777101*pi) q[25];
U1q(0.43272266737901*pi,1.5452340382573997*pi) q[26];
U1q(0.366040232901529*pi,1.0610002457923002*pi) q[27];
U1q(0.176513188354835*pi,1.1196065542984002*pi) q[28];
U1q(0.576850211889982*pi,0.9013440127470993*pi) q[29];
U1q(0.416763533511838*pi,1.4792959450954992*pi) q[30];
U1q(0.295312736158083*pi,1.6231648820923006*pi) q[31];
U1q(0.317218423305656*pi,1.0401950896775993*pi) q[32];
U1q(0.216375336078985*pi,1.8124515405822006*pi) q[33];
U1q(0.752411716855546*pi,0.06792084833110046*pi) q[34];
U1q(0.234811805702632*pi,0.1721652793351005*pi) q[35];
U1q(0.849588503250814*pi,1.9096540774103001*pi) q[36];
U1q(0.822957604371573*pi,1.0612051943165*pi) q[37];
U1q(0.598177713706487*pi,0.9377114448809998*pi) q[38];
U1q(0.808493679159507*pi,0.7521562975690408*pi) q[39];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[6],q[21];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[28],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[34],q[15];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[17],q[36];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[37],q[23];
RZZ(0.5*pi) q[27],q[25];
RZZ(0.5*pi) q[33],q[31];
U1q(0.354257401094795*pi,0.7794786664121993*pi) q[0];
U1q(0.477438626624972*pi,0.6345567324837003*pi) q[1];
U1q(0.683975765008277*pi,1.9982499731182006*pi) q[2];
U1q(0.330397782363933*pi,0.8522574990128007*pi) q[3];
U1q(0.595630607411247*pi,0.5955990151093999*pi) q[4];
U1q(0.6198755913226*pi,1.0413092156870007*pi) q[5];
U1q(0.534699574971018*pi,1.3156370078392996*pi) q[6];
U1q(0.515793428294387*pi,0.6002425095724995*pi) q[7];
U1q(0.577891422337515*pi,0.7327707199179194*pi) q[8];
U1q(0.238042507313141*pi,1.0163314042658005*pi) q[9];
U1q(0.326713236203489*pi,0.03851469400419916*pi) q[10];
U1q(0.0196278243791834*pi,1.5982006070002015*pi) q[11];
U1q(0.898820789588073*pi,1.6112415881885003*pi) q[12];
U1q(0.240769518963058*pi,0.10301548806929972*pi) q[13];
U1q(0.67932602515804*pi,0.6222832431654002*pi) q[14];
U1q(0.59242019615906*pi,1.3526604998089002*pi) q[15];
U1q(0.389188997776079*pi,0.2294324990704002*pi) q[16];
U1q(0.474845346435577*pi,1.6920112428826997*pi) q[17];
U1q(0.6525349161572*pi,1.2289347947150002*pi) q[18];
U1q(0.393145170703342*pi,1.2628963461518996*pi) q[19];
U1q(0.208190536104701*pi,0.09386679008109944*pi) q[20];
U1q(0.183027900146747*pi,1.4550228705692998*pi) q[21];
U1q(0.765739241316846*pi,1.8528307976370009*pi) q[22];
U1q(0.674271325308402*pi,0.6983217633866996*pi) q[23];
U1q(0.576143599301753*pi,1.1523081643476*pi) q[24];
U1q(0.666749994272883*pi,0.9880560620939702*pi) q[25];
U1q(0.679927314485789*pi,0.3542776825483003*pi) q[26];
U1q(0.471280062993331*pi,1.0541703335956996*pi) q[27];
U1q(0.435068960731502*pi,0.8259055302284999*pi) q[28];
U1q(0.722315745397966*pi,1.7005425707955002*pi) q[29];
U1q(0.800239109113521*pi,1.0796895952626002*pi) q[30];
U1q(0.784410686046701*pi,1.3331903004153993*pi) q[31];
U1q(0.651055450382489*pi,0.4507166680932002*pi) q[32];
U1q(0.414023517487518*pi,0.9951033082146985*pi) q[33];
U1q(0.104736630115191*pi,0.7919364973329994*pi) q[34];
U1q(0.651477598059313*pi,0.9907135191355003*pi) q[35];
U1q(0.62192953826066*pi,0.8414097380879006*pi) q[36];
U1q(0.64356910704068*pi,0.8870105977051992*pi) q[37];
U1q(0.386117658848056*pi,0.6856645339642*pi) q[38];
U1q(0.614371490903239*pi,1.4641519535325997*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[1],q[23];
RZZ(0.5*pi) q[7],q[2];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[4],q[14];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[29],q[6];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[16],q[12];
RZZ(0.5*pi) q[37],q[13];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[17],q[20];
RZZ(0.5*pi) q[34],q[18];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[33],q[21];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[32],q[27];
U1q(0.271661684356676*pi,1.2714261318028015*pi) q[0];
U1q(0.366273828798253*pi,1.6534597211554996*pi) q[1];
U1q(0.089989185583152*pi,0.18844481601770013*pi) q[2];
U1q(0.385063515713629*pi,1.2819184222152007*pi) q[3];
U1q(0.370581786448724*pi,1.8353083148140996*pi) q[4];
U1q(0.600252823671565*pi,1.3111024148533001*pi) q[5];
U1q(0.306052126526429*pi,1.6778333437597013*pi) q[6];
U1q(0.513801973949007*pi,0.2608874127419991*pi) q[7];
U1q(0.188711396779062*pi,1.8628494059977*pi) q[8];
U1q(0.311956786883179*pi,0.6720843660400995*pi) q[9];
U1q(0.690774401438944*pi,0.40571704807710063*pi) q[10];
U1q(0.258650408336769*pi,1.2548589782276984*pi) q[11];
U1q(0.526284317257827*pi,1.5070193145306003*pi) q[12];
U1q(0.0872262568039763*pi,1.2470077689605006*pi) q[13];
U1q(0.947597179699798*pi,0.5119922300230009*pi) q[14];
U1q(0.282646949519803*pi,0.0886178843877996*pi) q[15];
U1q(0.0651448112021512*pi,1.1911895914532007*pi) q[16];
U1q(0.699406215621057*pi,1.8278086818942008*pi) q[17];
U1q(0.459539192440212*pi,1.771286640638099*pi) q[18];
U1q(0.78863021913845*pi,0.16136155184680057*pi) q[19];
U1q(0.277381858802213*pi,0.3500366711596996*pi) q[20];
U1q(0.841293880320217*pi,0.41427590779139933*pi) q[21];
U1q(0.361233885885374*pi,0.47337205644300084*pi) q[22];
U1q(0.922691740228317*pi,0.6892230964472006*pi) q[23];
U1q(0.6505348033899*pi,0.21461653213230036*pi) q[24];
U1q(0.111503660800504*pi,0.10893651692946005*pi) q[25];
U1q(0.44349085226956*pi,0.7179293455230003*pi) q[26];
U1q(0.597387247839518*pi,1.3532169351903995*pi) q[27];
U1q(0.24235182576181*pi,0.26937237721769947*pi) q[28];
U1q(0.868988917062717*pi,0.2753172787314*pi) q[29];
U1q(0.717839457730994*pi,0.4109556065259987*pi) q[30];
U1q(0.896459219458367*pi,0.8160605890554002*pi) q[31];
U1q(0.672142669798531*pi,0.47171589371819955*pi) q[32];
U1q(0.550493379023262*pi,0.5795353576488012*pi) q[33];
U1q(0.777547975447773*pi,1.0623173034782987*pi) q[34];
U1q(0.153717893254054*pi,0.6706438604322997*pi) q[35];
U1q(0.636236848650022*pi,0.8978328368018005*pi) q[36];
U1q(0.594594286773072*pi,0.26252794203110064*pi) q[37];
U1q(0.742729645570808*pi,1.6409644033447996*pi) q[38];
U1q(0.342425555329771*pi,1.9435050035786006*pi) q[39];
RZZ(0.5*pi) q[19],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[25],q[2];
RZZ(0.5*pi) q[38],q[3];
RZZ(0.5*pi) q[4],q[28];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[11],q[9];
RZZ(0.5*pi) q[10],q[24];
RZZ(0.5*pi) q[13],q[12];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[15],q[22];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[35],q[21];
RZZ(0.5*pi) q[39],q[23];
RZZ(0.5*pi) q[29],q[32];
U1q(0.597434579186684*pi,0.22118180998599968*pi) q[0];
U1q(0.512265142537851*pi,0.2336212590942992*pi) q[1];
U1q(0.349416161375671*pi,1.6242860865209003*pi) q[2];
U1q(0.58562035942252*pi,1.7579165989975003*pi) q[3];
U1q(0.582126894542119*pi,0.8233391058499002*pi) q[4];
U1q(0.630462787114778*pi,0.9385648078947*pi) q[5];
U1q(0.575860845430777*pi,0.6253405402917984*pi) q[6];
U1q(0.522055040893535*pi,0.3236289953036007*pi) q[7];
U1q(0.609927208201726*pi,0.13105483121650074*pi) q[8];
U1q(0.847037396430636*pi,1.0192359876389006*pi) q[9];
U1q(0.427449887983953*pi,0.8943062890576989*pi) q[10];
U1q(0.734138418864822*pi,1.8982566101987004*pi) q[11];
U1q(0.345440639932886*pi,0.8525658590302001*pi) q[12];
U1q(0.358719521152929*pi,0.37535718772240045*pi) q[13];
U1q(0.249401112186338*pi,0.11914317729780066*pi) q[14];
U1q(0.367881911017484*pi,1.222978852582301*pi) q[15];
U1q(0.475631993080784*pi,1.2308671449467*pi) q[16];
U1q(0.766204198708656*pi,0.8729514260855993*pi) q[17];
U1q(0.671564695331209*pi,0.9002552947267013*pi) q[18];
U1q(0.270332736127922*pi,0.3835017947899999*pi) q[19];
U1q(0.299508115646205*pi,0.5749335774330007*pi) q[20];
U1q(0.88198600609333*pi,1.2018184716494993*pi) q[21];
U1q(0.512557419230006*pi,0.43134275360590024*pi) q[22];
U1q(0.262344101247826*pi,1.5417122772686014*pi) q[23];
U1q(0.736036703142517*pi,0.8263248057952985*pi) q[24];
U1q(0.11671766836069*pi,0.09670613022058028*pi) q[25];
U1q(0.674479207441402*pi,1.0399356262625012*pi) q[26];
U1q(0.578413809815656*pi,0.7155844844100994*pi) q[27];
U1q(0.763852706288383*pi,1.3358756267945004*pi) q[28];
U1q(0.523761577682724*pi,1.7985380109194011*pi) q[29];
U1q(0.356423217646094*pi,1.2476321428420007*pi) q[30];
U1q(0.701316931747217*pi,0.1314804560911007*pi) q[31];
U1q(0.356358632919146*pi,0.7688648951726993*pi) q[32];
U1q(0.758076595250313*pi,0.044034710508000785*pi) q[33];
U1q(0.587422754717071*pi,1.6517740490556*pi) q[34];
U1q(0.779116186780976*pi,0.06976817095100074*pi) q[35];
U1q(0.688402163966759*pi,1.6221922723956013*pi) q[36];
U1q(0.4296478056911*pi,0.9236230085273007*pi) q[37];
U1q(0.187952068516119*pi,1.4026424211523008*pi) q[38];
U1q(0.218586476566684*pi,1.2882736164151005*pi) q[39];
RZZ(0.5*pi) q[27],q[0];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[8],q[11];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[38],q[14];
RZZ(0.5*pi) q[26],q[15];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[17],q[19];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[32],q[21];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[34],q[33];
U1q(0.565213319095291*pi,1.7995379781231016*pi) q[0];
U1q(0.621630890280812*pi,0.1313131849627993*pi) q[1];
U1q(0.112303933739269*pi,1.6244858550693984*pi) q[2];
U1q(0.578102251649185*pi,0.15521367694400112*pi) q[3];
U1q(0.27852287267234*pi,1.5832127868699004*pi) q[4];
U1q(0.275742462223535*pi,1.0727733616089985*pi) q[5];
U1q(0.734697077223764*pi,0.5232896848204014*pi) q[6];
U1q(0.691833264476996*pi,1.0955047211987008*pi) q[7];
U1q(0.27782772276715*pi,1.1386441834902996*pi) q[8];
U1q(0.682209723130781*pi,1.5626131599582003*pi) q[9];
U1q(0.343396625294966*pi,0.27308837285820076*pi) q[10];
U1q(0.87809421043621*pi,1.8554642015973002*pi) q[11];
U1q(0.572614373130412*pi,0.9886405586005012*pi) q[12];
U1q(0.462444322303217*pi,0.6321453639626995*pi) q[13];
U1q(0.258469454841693*pi,0.9448120634468999*pi) q[14];
U1q(0.24720972189084*pi,1.3326556422535987*pi) q[15];
U1q(0.55940708459939*pi,0.7117665546480012*pi) q[16];
U1q(0.085874728143479*pi,0.22197693522570106*pi) q[17];
U1q(0.50158917892766*pi,1.552889136111201*pi) q[18];
U1q(0.154900997477403*pi,0.08998873047140066*pi) q[19];
U1q(0.346883638387585*pi,0.8061101245807016*pi) q[20];
U1q(0.823384043647584*pi,1.915064394450301*pi) q[21];
U1q(0.546862698711191*pi,1.7595983169187015*pi) q[22];
U1q(0.715131771422465*pi,0.30558140486320085*pi) q[23];
U1q(0.3736798510268*pi,0.9245134786260998*pi) q[24];
U1q(0.49734594741788*pi,1.6407928130520997*pi) q[25];
U1q(0.34211586421863*pi,0.5599450155788013*pi) q[26];
U1q(0.675704679591303*pi,1.3409986948076984*pi) q[27];
U1q(0.633500095900687*pi,0.3959345937830001*pi) q[28];
U1q(0.777619365735724*pi,0.06332715197849836*pi) q[29];
U1q(0.2470968895062*pi,1.0441180130652015*pi) q[30];
U1q(0.910823174728544*pi,1.8356464804049004*pi) q[31];
U1q(0.468575112187595*pi,0.7307165629846999*pi) q[32];
U1q(0.423347436035637*pi,0.3432746984646009*pi) q[33];
U1q(0.610459034306313*pi,0.8658709706703007*pi) q[34];
U1q(0.283630685959861*pi,0.4772534478071009*pi) q[35];
U1q(0.595800159570765*pi,1.947758194112101*pi) q[36];
U1q(0.202320254693615*pi,0.8856607772221992*pi) q[37];
U1q(0.38832174807843*pi,1.1036948257856984*pi) q[38];
U1q(0.773351641278603*pi,0.6158352811768992*pi) q[39];
RZZ(0.5*pi) q[6],q[0];
RZZ(0.5*pi) q[19],q[1];
RZZ(0.5*pi) q[36],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[32],q[4];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[10],q[14];
RZZ(0.5*pi) q[20],q[11];
RZZ(0.5*pi) q[12],q[23];
RZZ(0.5*pi) q[38],q[13];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[17],q[25];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[34],q[21];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[35],q[37];
U1q(0.901894447136623*pi,0.3448802699405995*pi) q[0];
U1q(0.741297596118296*pi,0.9891240431345985*pi) q[1];
U1q(0.645460816226619*pi,1.5672578598694002*pi) q[2];
U1q(0.264244620710115*pi,1.7540562118835012*pi) q[3];
U1q(0.629410089274902*pi,1.8392871048489*pi) q[4];
U1q(0.202436122284138*pi,0.31222740349389966*pi) q[5];
U1q(0.345437022208521*pi,1.9143685400425987*pi) q[6];
U1q(0.647021476825405*pi,1.7793381829714008*pi) q[7];
U1q(0.111244142777784*pi,0.18031314646690078*pi) q[8];
U1q(0.652207980665961*pi,0.4127377481730008*pi) q[9];
U1q(0.326450829499175*pi,1.519475978330199*pi) q[10];
U1q(0.740249195236274*pi,0.7482887403972995*pi) q[11];
U1q(0.157425142738812*pi,1.1912403468192991*pi) q[12];
U1q(0.467566095378918*pi,1.1290493025955008*pi) q[13];
U1q(0.751006230201443*pi,1.1434821421302992*pi) q[14];
U1q(0.720481190502284*pi,1.8608781811213007*pi) q[15];
U1q(0.437982361223953*pi,1.4196864325945988*pi) q[16];
U1q(0.634957420064588*pi,0.22149770071549924*pi) q[17];
U1q(0.579207945040524*pi,1.2666097749409992*pi) q[18];
U1q(0.676543194722559*pi,0.7771696140104005*pi) q[19];
U1q(0.756403141004503*pi,1.8991436182798012*pi) q[20];
U1q(0.458320639337094*pi,0.7634911906775983*pi) q[21];
U1q(0.894518839020111*pi,0.38467043542830126*pi) q[22];
U1q(0.847077336485738*pi,1.557725696355*pi) q[23];
U1q(0.122244945499191*pi,0.3361802952001014*pi) q[24];
U1q(0.39608376525836*pi,1.9066526373360997*pi) q[25];
U1q(0.926207511700606*pi,1.3853609015274984*pi) q[26];
U1q(0.711838230360487*pi,0.09488817009500039*pi) q[27];
U1q(0.133691959772658*pi,0.310245348645001*pi) q[28];
U1q(0.878560427650491*pi,1.3706482757332985*pi) q[29];
U1q(0.289967286727178*pi,1.4335369872264003*pi) q[30];
U1q(0.664461593083341*pi,0.8834456976378*pi) q[31];
U1q(0.196284427428643*pi,1.793475381082601*pi) q[32];
U1q(0.386406163790839*pi,1.2599630391258003*pi) q[33];
U1q(0.533490444721341*pi,1.1779481474571014*pi) q[34];
U1q(0.318665967895597*pi,0.24173269564040112*pi) q[35];
U1q(0.131559884306043*pi,1.1771220955556991*pi) q[36];
U1q(0.225538659039111*pi,0.548379895748301*pi) q[37];
U1q(0.877407863495482*pi,0.543028314112*pi) q[38];
U1q(0.207536839025497*pi,0.2038165500935989*pi) q[39];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[28],q[2];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[38],q[4];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[8],q[21];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[26],q[13];
RZZ(0.5*pi) q[14],q[23];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[16],q[39];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[37],q[19];
RZZ(0.5*pi) q[34],q[20];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[27],q[31];
RZZ(0.5*pi) q[29],q[30];
RZZ(0.5*pi) q[33],q[35];
U1q(0.659001975152172*pi,1.2652352738322996*pi) q[0];
U1q(0.842454642215954*pi,1.1678649569961017*pi) q[1];
U1q(0.48312522115135*pi,0.4889019691554992*pi) q[2];
U1q(0.634632126625278*pi,0.8329604844204006*pi) q[3];
U1q(0.308901760664939*pi,1.5180020941578007*pi) q[4];
U1q(0.63639742328405*pi,1.3197951696859*pi) q[5];
U1q(0.672900890156196*pi,0.5489609311091002*pi) q[6];
U1q(0.272480603243455*pi,0.2728989989718009*pi) q[7];
U1q(0.455991106059772*pi,1.354094820269701*pi) q[8];
U1q(0.830015142149756*pi,0.07847311638749943*pi) q[9];
U1q(0.361759756517402*pi,0.16082331844319953*pi) q[10];
U1q(0.440217055003131*pi,1.7811628504734003*pi) q[11];
U1q(0.264228271232328*pi,0.030614242690099047*pi) q[12];
U1q(0.36484610769548*pi,1.9977623864714005*pi) q[13];
U1q(0.741103226528319*pi,0.4663402563830985*pi) q[14];
U1q(0.266871205501138*pi,1.7294024013087999*pi) q[15];
U1q(0.442136152067099*pi,0.6446146530142016*pi) q[16];
U1q(0.498317972313619*pi,1.4844407085001983*pi) q[17];
U1q(0.767591577478523*pi,0.23815974973240017*pi) q[18];
U1q(0.440727798301704*pi,1.0760587391488983*pi) q[19];
U1q(0.537232940265867*pi,1.2103798063358013*pi) q[20];
U1q(0.510501733521845*pi,0.7616502591652008*pi) q[21];
U1q(0.285535510803232*pi,1.9465154973146994*pi) q[22];
U1q(0.558991395918152*pi,0.8954111959005004*pi) q[23];
U1q(0.359160504876846*pi,0.23061638700459852*pi) q[24];
U1q(0.551374465114245*pi,1.5047116585138003*pi) q[25];
U1q(0.953608831159154*pi,1.7719924659243986*pi) q[26];
U1q(0.494299284190032*pi,1.9482486305372007*pi) q[27];
U1q(0.860167017573854*pi,1.0715664366834012*pi) q[28];
U1q(0.429129836541667*pi,0.2378933788982991*pi) q[29];
U1q(0.224015065935458*pi,1.1440238002652983*pi) q[30];
U1q(0.575515102223556*pi,1.9085703908210014*pi) q[31];
U1q(0.685622375977418*pi,1.5779440363311004*pi) q[32];
U1q(0.565699279023092*pi,0.8968635625799983*pi) q[33];
U1q(0.955851611708149*pi,1.3967860211433987*pi) q[34];
U1q(0.695187856483989*pi,0.5306880618900003*pi) q[35];
U1q(0.0769462757924603*pi,1.5352956538647007*pi) q[36];
U1q(0.489920436270883*pi,0.17119719095520125*pi) q[37];
U1q(0.298648632304148*pi,0.41853433923250094*pi) q[38];
U1q(0.282319608215881*pi,0.295901701331001*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[35],q[2];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[4],q[13];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[9];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[28],q[11];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[16],q[27];
RZZ(0.5*pi) q[29],q[18];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[25],q[20];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[34],q[26];
RZZ(0.5*pi) q[32],q[31];
U1q(0.319616985861707*pi,0.7106459777825975*pi) q[0];
U1q(0.645784337224271*pi,0.9490724296246*pi) q[1];
U1q(0.864185154175279*pi,0.3537649828063003*pi) q[2];
U1q(0.939022975685562*pi,0.8083376762577998*pi) q[3];
U1q(0.520647253210917*pi,0.21394606599299948*pi) q[4];
U1q(0.729792090450902*pi,1.2789136681970987*pi) q[5];
U1q(0.31742369520259*pi,0.37240670696689904*pi) q[6];
U1q(0.83284037716031*pi,0.6502674153177992*pi) q[7];
U1q(0.799114918537508*pi,1.3319515740554984*pi) q[8];
U1q(0.267378404045698*pi,0.7479916375258*pi) q[9];
U1q(0.525384751264327*pi,1.8675542173551989*pi) q[10];
U1q(0.35090515366184*pi,1.9096230690248994*pi) q[11];
U1q(0.160432106513462*pi,1.4549298122608008*pi) q[12];
U1q(0.747201555094469*pi,0.013369961809001296*pi) q[13];
U1q(0.352632839806934*pi,0.2640070339834004*pi) q[14];
U1q(0.816760741619615*pi,0.30368180279980095*pi) q[15];
U1q(0.518141012011924*pi,1.8066779084050992*pi) q[16];
U1q(0.766850152296746*pi,1.2113022452598017*pi) q[17];
U1q(0.628212184178766*pi,1.2015729088310003*pi) q[18];
U1q(0.537363663342691*pi,1.3441095134904018*pi) q[19];
U1q(0.707495347966241*pi,1.0662076648072016*pi) q[20];
U1q(0.54861794427391*pi,1.9839179722084985*pi) q[21];
U1q(0.670703828951145*pi,1.866403476018501*pi) q[22];
U1q(0.380609122137409*pi,1.6924277526557994*pi) q[23];
U1q(0.248708970057955*pi,1.2917938598275995*pi) q[24];
U1q(0.187789108926451*pi,0.6136196932236011*pi) q[25];
U1q(0.591547606472824*pi,0.32709555970609827*pi) q[26];
U1q(0.473609199346862*pi,1.590255635368301*pi) q[27];
U1q(0.759394320168499*pi,0.5841359334378993*pi) q[28];
U1q(0.45793186795368*pi,0.6529404931295986*pi) q[29];
U1q(0.335788024937407*pi,1.5860888626184995*pi) q[30];
U1q(0.245741272054677*pi,1.6073130846969015*pi) q[31];
U1q(0.792669618645506*pi,1.1036212590604002*pi) q[32];
U1q(0.474068674976667*pi,0.030454682137001754*pi) q[33];
U1q(0.596018476023778*pi,0.38850190928589967*pi) q[34];
U1q(0.0788341812662168*pi,1.9621437707775016*pi) q[35];
U1q(0.693848193521275*pi,1.7480686166518993*pi) q[36];
U1q(0.288265866014377*pi,1.0920141716444007*pi) q[37];
U1q(0.307470304007736*pi,0.5482291326780988*pi) q[38];
U1q(0.369748830561818*pi,1.0017739756396011*pi) q[39];
rz(1.4853019872659985*pi) q[0];
rz(2.8524393221495004*pi) q[1];
rz(3.888769626596499*pi) q[2];
rz(1.6198648125608983*pi) q[3];
rz(3.9296443590004984*pi) q[4];
rz(3.0086223914649004*pi) q[5];
rz(0.5789075491037998*pi) q[6];
rz(1.9570805891858*pi) q[7];
rz(2.259111733386799*pi) q[8];
rz(2.5847937706856*pi) q[9];
rz(2.3375048598072006*pi) q[10];
rz(1.5874932257355994*pi) q[11];
rz(1.8020323015608*pi) q[12];
rz(2.7472667411635996*pi) q[13];
rz(2.6538232501077985*pi) q[14];
rz(2.6799026934849017*pi) q[15];
rz(0.3395324760422014*pi) q[16];
rz(0.4222061357951006*pi) q[17];
rz(2.505592859456499*pi) q[18];
rz(0.17438925137570038*pi) q[19];
rz(0.15471728491069925*pi) q[20];
rz(0.07790342196989997*pi) q[21];
rz(1.0073968600854997*pi) q[22];
rz(0.7606352235964984*pi) q[23];
rz(2.769308803158701*pi) q[24];
rz(0.4901432821509992*pi) q[25];
rz(1.1514530103521992*pi) q[26];
rz(3.1017405650087007*pi) q[27];
rz(3.060692533884499*pi) q[28];
rz(2.8757175734484015*pi) q[29];
rz(1.6511632111125998*pi) q[30];
rz(2.1288069992923013*pi) q[31];
rz(1.8360026974267996*pi) q[32];
rz(3.7691715183709995*pi) q[33];
rz(3.6703918742273984*pi) q[34];
rz(0.21424636362180038*pi) q[35];
rz(2.683830434781399*pi) q[36];
rz(2.7401243749610984*pi) q[37];
rz(2.527989273296299*pi) q[38];
rz(3.2628458568091006*pi) q[39];
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