OPENQASM 2.0;
include "hqslib1.inc";

qreg q[56];
creg c[56];
U1q(0.886955248009186*pi,1.317857288863404*pi) q[0];
U1q(0.623264776644037*pi,0.920630923999287*pi) q[1];
U1q(0.555864356279936*pi,0.318150485879523*pi) q[2];
U1q(0.740180551743352*pi,0.533698401166734*pi) q[3];
U1q(0.608451249689222*pi,1.789983347637027*pi) q[4];
U1q(0.440261631458475*pi,1.249974424310375*pi) q[5];
U1q(0.346137342311117*pi,0.801467507554299*pi) q[6];
U1q(0.565120403814881*pi,1.588776818976867*pi) q[7];
U1q(0.363433290407057*pi,0.292889305344179*pi) q[8];
U1q(0.207095654705288*pi,0.781571699349222*pi) q[9];
U1q(0.25692322763082*pi,1.096153794302308*pi) q[10];
U1q(0.492697117626643*pi,1.1386072850210112*pi) q[11];
U1q(0.710686201215113*pi,0.075001937293181*pi) q[12];
U1q(0.739957602873736*pi,0.333522762149996*pi) q[13];
U1q(0.311730344198284*pi,1.371667961136911*pi) q[14];
U1q(0.702087947273447*pi,0.00951350192960887*pi) q[15];
U1q(0.801968680584628*pi,0.45025949371756*pi) q[16];
U1q(0.721180952740227*pi,1.164790272802231*pi) q[17];
U1q(0.40875953204816*pi,0.0680918907629238*pi) q[18];
U1q(0.453159462525231*pi,1.569655265730488*pi) q[19];
U1q(0.529776696546566*pi,1.25462539082193*pi) q[20];
U1q(0.699376108386318*pi,0.536402272584491*pi) q[21];
U1q(0.77098353860148*pi,0.523147344701385*pi) q[22];
U1q(0.598956004760621*pi,1.50658442631198*pi) q[23];
U1q(0.491760127270591*pi,0.238497025647012*pi) q[24];
U1q(0.890907407901573*pi,1.73640960953007*pi) q[25];
U1q(0.422301568309984*pi,0.189918201142507*pi) q[26];
U1q(0.456722211715129*pi,0.259946299347782*pi) q[27];
U1q(0.937905635789237*pi,1.345903445440186*pi) q[28];
U1q(0.931121094342129*pi,0.836209078384376*pi) q[29];
U1q(0.450679185816291*pi,0.806096552907628*pi) q[30];
U1q(0.608406647767918*pi,1.02371859659807*pi) q[31];
U1q(0.37474194660433*pi,0.399995136402859*pi) q[32];
U1q(0.214266513518568*pi,0.81279313771942*pi) q[33];
U1q(0.37192518098387*pi,0.425913469943145*pi) q[34];
U1q(0.77586159019154*pi,1.2814885873323*pi) q[35];
U1q(0.295656426351695*pi,0.188914638516967*pi) q[36];
U1q(0.52003879559847*pi,0.519562258607591*pi) q[37];
U1q(0.249302472462761*pi,1.830155940949789*pi) q[38];
U1q(0.335954466395699*pi,1.318690156252996*pi) q[39];
U1q(0.203210764385949*pi,0.181332676673102*pi) q[40];
U1q(0.765709499910241*pi,1.390700820087766*pi) q[41];
U1q(0.22835156098076*pi,1.383334510175722*pi) q[42];
U1q(0.468452974057143*pi,0.448347055286362*pi) q[43];
U1q(0.27413488615424*pi,1.619582480411395*pi) q[44];
U1q(0.975030890042805*pi,0.665600451747247*pi) q[45];
U1q(0.308598415388314*pi,0.015032160878795*pi) q[46];
U1q(0.59278185436407*pi,0.0610113267190602*pi) q[47];
U1q(0.433204998123483*pi,1.669131319627923*pi) q[48];
U1q(0.934702143850862*pi,0.153413065286408*pi) q[49];
U1q(0.759612792320888*pi,0.165556332036522*pi) q[50];
U1q(0.152421994774179*pi,1.790599374476624*pi) q[51];
U1q(0.445405315497634*pi,0.206974216815659*pi) q[52];
U1q(0.332851804582658*pi,1.0807520743049261*pi) q[53];
U1q(0.745434592172181*pi,1.17478422676726*pi) q[54];
U1q(0.236328360753193*pi,0.231203549853585*pi) q[55];
RZZ(0.5*pi) q[3],q[0];
RZZ(0.5*pi) q[52],q[1];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[43];
RZZ(0.5*pi) q[8],q[29];
RZZ(0.5*pi) q[22],q[9];
RZZ(0.5*pi) q[46],q[10];
RZZ(0.5*pi) q[11],q[47];
RZZ(0.5*pi) q[21],q[12];
RZZ(0.5*pi) q[13],q[35];
RZZ(0.5*pi) q[14],q[44];
RZZ(0.5*pi) q[15],q[42];
RZZ(0.5*pi) q[53],q[16];
RZZ(0.5*pi) q[50],q[18];
RZZ(0.5*pi) q[37],q[19];
RZZ(0.5*pi) q[41],q[20];
RZZ(0.5*pi) q[23],q[28];
RZZ(0.5*pi) q[24],q[55];
RZZ(0.5*pi) q[32],q[25];
RZZ(0.5*pi) q[34],q[26];
RZZ(0.5*pi) q[27],q[36];
RZZ(0.5*pi) q[33],q[54];
RZZ(0.5*pi) q[45],q[38];
RZZ(0.5*pi) q[48],q[40];
RZZ(0.5*pi) q[51],q[49];
U1q(0.890902494439135*pi,0.2635247471052402*pi) q[0];
U1q(0.666610798475452*pi,1.1919792109785181*pi) q[1];
U1q(0.553504600051249*pi,0.248249469271362*pi) q[2];
U1q(0.36083499765496*pi,0.27232641277048986*pi) q[3];
U1q(0.347388873471696*pi,0.3095224483401098*pi) q[4];
U1q(0.333615613032215*pi,0.9842952415500399*pi) q[5];
U1q(0.193461909960575*pi,0.09814326179149013*pi) q[6];
U1q(0.475748682786805*pi,0.8415877969679699*pi) q[7];
U1q(0.429038030909468*pi,0.09464026468660003*pi) q[8];
U1q(0.872038434406983*pi,0.80751475310912*pi) q[9];
U1q(0.561655440889873*pi,0.9829034543061201*pi) q[10];
U1q(0.37238911056842*pi,1.0145924639111499*pi) q[11];
U1q(0.865708895240204*pi,0.9421187764042198*pi) q[12];
U1q(0.514585840081079*pi,0.7436154900154199*pi) q[13];
U1q(0.814212592321515*pi,0.030279101543710052*pi) q[14];
U1q(0.45611971761676*pi,1.63773536660468*pi) q[15];
U1q(0.44548947186304*pi,1.131772646580218*pi) q[16];
U1q(0.761168894353807*pi,0.65826130592691*pi) q[17];
U1q(0.522826427508922*pi,0.4393892768163701*pi) q[18];
U1q(0.903209824842971*pi,0.15368720821911008*pi) q[19];
U1q(0.26593980751116*pi,0.432131882794536*pi) q[20];
U1q(0.34839234919238*pi,0.3790060486676601*pi) q[21];
U1q(0.519522883683733*pi,0.65643580903557*pi) q[22];
U1q(0.569914071727629*pi,0.0777574715272881*pi) q[23];
U1q(0.0872999007006039*pi,0.4067668661617301*pi) q[24];
U1q(0.565731672170337*pi,0.6185211883748698*pi) q[25];
U1q(0.4027748665147*pi,1.12746096684611*pi) q[26];
U1q(0.371496269287308*pi,0.04749789359440992*pi) q[27];
U1q(0.690570671028934*pi,0.1561133915927302*pi) q[28];
U1q(0.409231180417742*pi,1.157124332519827*pi) q[29];
U1q(0.582362200443266*pi,1.019477258604478*pi) q[30];
U1q(0.591360952189634*pi,0.93019683738203*pi) q[31];
U1q(0.466165460530853*pi,1.9414726667881*pi) q[32];
U1q(0.746401299191022*pi,1.8122024386772901*pi) q[33];
U1q(0.296984738831942*pi,1.76366689231362*pi) q[34];
U1q(0.645502775821192*pi,0.85114038159281*pi) q[35];
U1q(0.351345740046279*pi,0.92987173995674*pi) q[36];
U1q(0.713998849695382*pi,0.3930351661485201*pi) q[37];
U1q(0.559072880958993*pi,0.022615572858819988*pi) q[38];
U1q(0.206712041290908*pi,1.52428531523059*pi) q[39];
U1q(0.220729506082391*pi,0.26880588332880007*pi) q[40];
U1q(0.585606201199085*pi,1.7507225074969304*pi) q[41];
U1q(0.361917200596238*pi,0.2491976787228598*pi) q[42];
U1q(0.216465117630229*pi,1.3226861699350998*pi) q[43];
U1q(0.61468518807343*pi,0.69868282270102*pi) q[44];
U1q(0.756869244149723*pi,0.4864714049579999*pi) q[45];
U1q(0.365384658670611*pi,0.3122426651922998*pi) q[46];
U1q(0.0535688248490048*pi,0.8494837028469899*pi) q[47];
U1q(0.275738139093467*pi,1.91215483126222*pi) q[48];
U1q(0.543855121246854*pi,1.7348745619499102*pi) q[49];
U1q(0.807868855681624*pi,1.4967217069728602*pi) q[50];
U1q(0.595005552643368*pi,1.71599224081166*pi) q[51];
U1q(0.660451248173519*pi,1.9568351120233598*pi) q[52];
U1q(0.345169788411422*pi,0.17552777803655983*pi) q[53];
U1q(0.490677734731514*pi,0.888576036720498*pi) q[54];
U1q(0.252474260417341*pi,1.32466654947644*pi) q[55];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[2],q[51];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[4],q[17];
RZZ(0.5*pi) q[5],q[7];
RZZ(0.5*pi) q[13],q[6];
RZZ(0.5*pi) q[8],q[53];
RZZ(0.5*pi) q[19],q[9];
RZZ(0.5*pi) q[37],q[10];
RZZ(0.5*pi) q[11],q[42];
RZZ(0.5*pi) q[52],q[12];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[15],q[23];
RZZ(0.5*pi) q[39],q[16];
RZZ(0.5*pi) q[18],q[40];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[21],q[49];
RZZ(0.5*pi) q[22],q[28];
RZZ(0.5*pi) q[24],q[29];
RZZ(0.5*pi) q[25],q[47];
RZZ(0.5*pi) q[48],q[26];
RZZ(0.5*pi) q[54],q[27];
RZZ(0.5*pi) q[50],q[30];
RZZ(0.5*pi) q[33],q[44];
RZZ(0.5*pi) q[41],q[38];
RZZ(0.5*pi) q[55],q[43];
RZZ(0.5*pi) q[46],q[45];
U1q(0.300394341803669*pi,0.019973063485109854*pi) q[0];
U1q(0.495191553594104*pi,1.47352475177845*pi) q[1];
U1q(0.372030243536723*pi,1.402787127278886*pi) q[2];
U1q(0.539678679114862*pi,1.22602507571259*pi) q[3];
U1q(0.279475408410273*pi,1.7603422021912696*pi) q[4];
U1q(0.219092604063284*pi,1.8404055265657302*pi) q[5];
U1q(0.388777383266303*pi,1.4052200713978804*pi) q[6];
U1q(0.612319996794783*pi,1.4571117159545697*pi) q[7];
U1q(0.39905040123837*pi,0.23854038732442007*pi) q[8];
U1q(0.107552953664058*pi,0.58811632712845*pi) q[9];
U1q(0.26160677519533*pi,0.3896496435847503*pi) q[10];
U1q(0.535972600934675*pi,1.0069730157402903*pi) q[11];
U1q(0.545929146336932*pi,1.25526050715124*pi) q[12];
U1q(0.260307354560017*pi,1.57817037131041*pi) q[13];
U1q(0.606576705587507*pi,1.28808374950933*pi) q[14];
U1q(0.617478096466507*pi,0.4386035043121197*pi) q[15];
U1q(0.130250216435245*pi,0.29655909764076016*pi) q[16];
U1q(0.722723347916055*pi,1.8901427164622397*pi) q[17];
U1q(0.490292134691858*pi,0.9698957874377498*pi) q[18];
U1q(0.531877058724006*pi,1.9882457317992301*pi) q[19];
U1q(0.586559706877636*pi,1.18135526507412*pi) q[20];
U1q(0.763373949344667*pi,1.7413381109431603*pi) q[21];
U1q(0.563176908166794*pi,0.5761204731573697*pi) q[22];
U1q(0.637527471949455*pi,1.7711805305030102*pi) q[23];
U1q(0.764369012317733*pi,0.029710577982239972*pi) q[24];
U1q(0.285658375259602*pi,1.17633697683722*pi) q[25];
U1q(0.227589488264716*pi,1.67805668587875*pi) q[26];
U1q(0.648285851327132*pi,1.8601542013110697*pi) q[27];
U1q(0.594395652352062*pi,0.19547308378646022*pi) q[28];
U1q(0.85785867616768*pi,1.63584566653229*pi) q[29];
U1q(0.673618450492249*pi,0.020653579040349923*pi) q[30];
U1q(0.583999716778409*pi,1.40423973620805*pi) q[31];
U1q(0.0400311191202798*pi,1.9223355109580798*pi) q[32];
U1q(0.75442870864637*pi,0.28077794953040014*pi) q[33];
U1q(0.251002371216571*pi,1.6693789691256997*pi) q[34];
U1q(0.577273208995455*pi,1.95029678930287*pi) q[35];
U1q(0.253078722766094*pi,1.8156878238330103*pi) q[36];
U1q(0.726594389955152*pi,0.0610496780002201*pi) q[37];
U1q(0.470084991078795*pi,1.7759658555944302*pi) q[38];
U1q(0.600270095485163*pi,0.99817397690442*pi) q[39];
U1q(0.680457984302218*pi,1.2750298998349097*pi) q[40];
U1q(0.349294798775634*pi,0.8383558338591*pi) q[41];
U1q(0.394162090367021*pi,0.75608626589315*pi) q[42];
U1q(0.443140569972264*pi,1.41459355643181*pi) q[43];
U1q(0.868408890625752*pi,0.5170952404829401*pi) q[44];
U1q(0.386386226981348*pi,1.4663410118554197*pi) q[45];
U1q(0.349357322372439*pi,0.6635476786683698*pi) q[46];
U1q(0.488121344979594*pi,1.1939862449663403*pi) q[47];
U1q(0.757230896781923*pi,1.1108868822132303*pi) q[48];
U1q(0.757763352023835*pi,1.8069421006209199*pi) q[49];
U1q(0.675489193708612*pi,1.1565219560676798*pi) q[50];
U1q(0.603556171882972*pi,1.3035962741727598*pi) q[51];
U1q(0.436517967745212*pi,1.08747769002429*pi) q[52];
U1q(0.327910382763125*pi,0.8381351311315601*pi) q[53];
U1q(0.634426260798191*pi,1.4713057614947451*pi) q[54];
U1q(0.190085548636539*pi,1.4881747614336902*pi) q[55];
RZZ(0.5*pi) q[0],q[55];
RZZ(0.5*pi) q[1],q[39];
RZZ(0.5*pi) q[2],q[18];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[48],q[4];
RZZ(0.5*pi) q[5],q[27];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[11],q[9];
RZZ(0.5*pi) q[10],q[45];
RZZ(0.5*pi) q[25],q[12];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[14],q[42];
RZZ(0.5*pi) q[33],q[17];
RZZ(0.5*pi) q[19],q[51];
RZZ(0.5*pi) q[52],q[20];
RZZ(0.5*pi) q[21],q[40];
RZZ(0.5*pi) q[22],q[44];
RZZ(0.5*pi) q[23],q[47];
RZZ(0.5*pi) q[35],q[24];
RZZ(0.5*pi) q[32],q[26];
RZZ(0.5*pi) q[28],q[53];
RZZ(0.5*pi) q[54],q[29];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[50],q[31];
RZZ(0.5*pi) q[46],q[34];
RZZ(0.5*pi) q[37],q[49];
RZZ(0.5*pi) q[41],q[43];
U1q(0.426254253693753*pi,1.7681890257361008*pi) q[0];
U1q(0.206558475355532*pi,1.5051131124239596*pi) q[1];
U1q(0.131926557188692*pi,1.42540026444382*pi) q[2];
U1q(0.40848310954702*pi,1.1069600254066296*pi) q[3];
U1q(0.410506586553389*pi,1.9062395034751596*pi) q[4];
U1q(0.564732763089885*pi,0.99708198209012*pi) q[5];
U1q(0.455077181902814*pi,0.8479724635963404*pi) q[6];
U1q(0.647552545062713*pi,0.6470023792548698*pi) q[7];
U1q(0.67522967073658*pi,1.7717936915111503*pi) q[8];
U1q(0.342896943460538*pi,0.7642820150986704*pi) q[9];
U1q(0.909882177580618*pi,0.26816646888056006*pi) q[10];
U1q(0.752909581056548*pi,1.96478128261586*pi) q[11];
U1q(0.717649586810423*pi,1.0055905317267708*pi) q[12];
U1q(0.610324313427962*pi,0.9558760262761696*pi) q[13];
U1q(0.735085556367738*pi,1.8588854424921202*pi) q[14];
U1q(0.686048302178733*pi,1.2105427119413*pi) q[15];
U1q(0.425810149441518*pi,0.034618024076659815*pi) q[16];
U1q(0.498204605579968*pi,1.51001552754959*pi) q[17];
U1q(0.68889788683235*pi,1.8554489026667804*pi) q[18];
U1q(0.391730297765302*pi,0.5499756183960303*pi) q[19];
U1q(0.4836768780068*pi,0.7319859776011199*pi) q[20];
U1q(0.305991837608246*pi,1.1893179566713599*pi) q[21];
U1q(0.628679561944358*pi,1.9494699329064993*pi) q[22];
U1q(0.503586111424789*pi,0.25900335068124036*pi) q[23];
U1q(0.574073346751742*pi,1.6883363547594596*pi) q[24];
U1q(0.656109490013457*pi,0.8180498472770203*pi) q[25];
U1q(0.403892195320784*pi,0.06589339382644077*pi) q[26];
U1q(0.543919621012281*pi,1.4888250781022805*pi) q[27];
U1q(0.238954083834371*pi,1.9140574790279992*pi) q[28];
U1q(0.673368616341503*pi,0.9831487185960999*pi) q[29];
U1q(0.42224750273439*pi,1.73551285616005*pi) q[30];
U1q(0.103522236359176*pi,0.5974540214171*pi) q[31];
U1q(0.487397547955788*pi,1.3415281498596592*pi) q[32];
U1q(0.850205223218363*pi,1.2529922598571304*pi) q[33];
U1q(0.648199809006479*pi,1.9305567112111603*pi) q[34];
U1q(0.267465822503126*pi,0.2641465488749599*pi) q[35];
U1q(0.257152685914992*pi,1.7327717555851496*pi) q[36];
U1q(0.785121268576731*pi,0.13173074447210986*pi) q[37];
U1q(0.478790405398555*pi,1.7670224906425904*pi) q[38];
U1q(0.182066115121572*pi,1.1956428424948298*pi) q[39];
U1q(0.553722468796211*pi,1.8953093284063591*pi) q[40];
U1q(0.581334123561719*pi,0.04298331201231953*pi) q[41];
U1q(0.256949662569965*pi,0.8303855223844501*pi) q[42];
U1q(0.634259826620143*pi,0.36694652722021015*pi) q[43];
U1q(0.691194719959042*pi,0.8893365001491196*pi) q[44];
U1q(0.203418590694223*pi,0.8202300540099401*pi) q[45];
U1q(0.404257896175958*pi,0.9634396755396502*pi) q[46];
U1q(0.791630411004055*pi,0.6188251166204601*pi) q[47];
U1q(0.463918616662581*pi,0.5099988204764898*pi) q[48];
U1q(0.55258709469367*pi,1.6837593071584003*pi) q[49];
U1q(0.555859185666258*pi,0.3353552849155603*pi) q[50];
U1q(0.28515146834725*pi,0.8738945102609099*pi) q[51];
U1q(0.227137801247678*pi,1.6977823237067202*pi) q[52];
U1q(0.384351364364645*pi,1.9486572944477007*pi) q[53];
U1q(0.529365920972499*pi,0.19470835619418003*pi) q[54];
U1q(0.449582472403128*pi,1.7370703668158392*pi) q[55];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[1],q[11];
RZZ(0.5*pi) q[2],q[53];
RZZ(0.5*pi) q[46],q[3];
RZZ(0.5*pi) q[4],q[41];
RZZ(0.5*pi) q[31],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[54],q[7];
RZZ(0.5*pi) q[8],q[20];
RZZ(0.5*pi) q[51],q[9];
RZZ(0.5*pi) q[10],q[40];
RZZ(0.5*pi) q[13],q[12];
RZZ(0.5*pi) q[14],q[23];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[33],q[16];
RZZ(0.5*pi) q[17],q[43];
RZZ(0.5*pi) q[34],q[18];
RZZ(0.5*pi) q[19],q[30];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[25],q[48];
RZZ(0.5*pi) q[27],q[47];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[42],q[29];
RZZ(0.5*pi) q[44],q[36];
RZZ(0.5*pi) q[52],q[39];
RZZ(0.5*pi) q[45],q[49];
RZZ(0.5*pi) q[50],q[55];
U1q(0.726434357379434*pi,1.4784679016606006*pi) q[0];
U1q(0.780473288406819*pi,0.013299956117600154*pi) q[1];
U1q(0.499833586533554*pi,1.4075754274863899*pi) q[2];
U1q(0.785907870383123*pi,0.78786997609234*pi) q[3];
U1q(0.623607952764315*pi,1.2125217278513993*pi) q[4];
U1q(0.390425693843377*pi,1.5495515986937303*pi) q[5];
U1q(0.67565838384789*pi,0.09717021228789946*pi) q[6];
U1q(0.610755314227587*pi,0.06460110631740079*pi) q[7];
U1q(0.369360219437506*pi,1.5675775550603994*pi) q[8];
U1q(0.213237541327364*pi,1.0749128438313402*pi) q[9];
U1q(0.555628909939832*pi,0.8513702780835395*pi) q[10];
U1q(0.763744948412176*pi,0.4543728849813*pi) q[11];
U1q(0.491658511169696*pi,0.7104109953529001*pi) q[12];
U1q(0.24020355831509*pi,1.4527974708041*pi) q[13];
U1q(0.372160432071824*pi,0.5742139800230204*pi) q[14];
U1q(0.255558822817863*pi,1.5026243062934999*pi) q[15];
U1q(0.777209110668985*pi,0.09382972040369975*pi) q[16];
U1q(0.5427071912549*pi,0.5572965381734694*pi) q[17];
U1q(0.326958063574267*pi,0.5821238322247702*pi) q[18];
U1q(0.536171790309021*pi,1.1599689999123708*pi) q[19];
U1q(0.707390656824809*pi,0.34132193368628005*pi) q[20];
U1q(0.528882140739568*pi,1.0526504706151503*pi) q[21];
U1q(0.745330981142653*pi,0.7554210634549001*pi) q[22];
U1q(0.644343288430568*pi,0.8037945596465299*pi) q[23];
U1q(0.33310316400391*pi,1.3644118312574598*pi) q[24];
U1q(0.134875106887877*pi,0.26664829611280005*pi) q[25];
U1q(0.938223528855573*pi,0.9292584137568998*pi) q[26];
U1q(0.611987612676426*pi,0.2005010125831692*pi) q[27];
U1q(0.0765882517626867*pi,0.6349996028720994*pi) q[28];
U1q(0.501817488717296*pi,0.1801971612801303*pi) q[29];
U1q(0.345535266950868*pi,1.1129636565568797*pi) q[30];
U1q(0.553376008591125*pi,1.6785653774577396*pi) q[31];
U1q(0.336473682707116*pi,1.5137660626902*pi) q[32];
U1q(0.395138934856747*pi,1.0779800302983809*pi) q[33];
U1q(0.117982066406539*pi,1.4972821091503992*pi) q[34];
U1q(0.797490820087676*pi,0.8889683300076703*pi) q[35];
U1q(0.458321195636748*pi,0.8136518295256305*pi) q[36];
U1q(0.423123782494047*pi,0.12304420001897043*pi) q[37];
U1q(0.429875131230364*pi,1.4990835771516995*pi) q[38];
U1q(0.521097188519009*pi,1.9984755191799994*pi) q[39];
U1q(0.443285495088731*pi,0.1327935414212007*pi) q[40];
U1q(0.337379690899596*pi,0.5302928665077005*pi) q[41];
U1q(0.795669904722672*pi,0.4282669580827001*pi) q[42];
U1q(0.424740739732439*pi,1.9303847320148098*pi) q[43];
U1q(0.357852639490139*pi,1.0812383399153198*pi) q[44];
U1q(0.753853668857416*pi,1.7675724126504004*pi) q[45];
U1q(0.362260957362283*pi,1.7992856410615001*pi) q[46];
U1q(0.381803354845582*pi,1.9949902728281508*pi) q[47];
U1q(0.61333445047995*pi,1.7746070360581996*pi) q[48];
U1q(0.387208844305097*pi,0.8922535554406004*pi) q[49];
U1q(0.464893376163547*pi,0.5152098867791004*pi) q[50];
U1q(0.175258415194443*pi,1.3702778231448498*pi) q[51];
U1q(0.460943976327133*pi,1.3459864384577003*pi) q[52];
U1q(0.421126839912413*pi,0.0779863841923003*pi) q[53];
U1q(0.420825420033865*pi,0.6246664779050697*pi) q[54];
U1q(0.703584881862866*pi,0.7302091359249605*pi) q[55];
RZZ(0.5*pi) q[54],q[0];
RZZ(0.5*pi) q[1],q[30];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[3],q[45];
RZZ(0.5*pi) q[8],q[4];
RZZ(0.5*pi) q[5],q[29];
RZZ(0.5*pi) q[6],q[42];
RZZ(0.5*pi) q[19],q[7];
RZZ(0.5*pi) q[31],q[9];
RZZ(0.5*pi) q[44],q[10];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[12],q[39];
RZZ(0.5*pi) q[46],q[13];
RZZ(0.5*pi) q[15],q[24];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[22],q[18];
RZZ(0.5*pi) q[43],q[20];
RZZ(0.5*pi) q[21],q[47];
RZZ(0.5*pi) q[23],q[55];
RZZ(0.5*pi) q[25],q[40];
RZZ(0.5*pi) q[35],q[26];
RZZ(0.5*pi) q[27],q[34];
RZZ(0.5*pi) q[32],q[37];
RZZ(0.5*pi) q[33],q[41];
RZZ(0.5*pi) q[36],q[49];
RZZ(0.5*pi) q[52],q[38];
RZZ(0.5*pi) q[50],q[48];
RZZ(0.5*pi) q[51],q[53];
U1q(0.578862548246238*pi,0.5637417948635992*pi) q[0];
U1q(0.612067409001154*pi,0.04157966892570997*pi) q[1];
U1q(0.702220914204332*pi,1.18655012809755*pi) q[2];
U1q(0.131133194522282*pi,0.025230380167100108*pi) q[3];
U1q(0.390658502299906*pi,0.6900373229817003*pi) q[4];
U1q(0.756198954819392*pi,0.14966601245592948*pi) q[5];
U1q(0.268747414603491*pi,1.0991845629720984*pi) q[6];
U1q(0.428666620440493*pi,1.4338735381441001*pi) q[7];
U1q(0.559075248143511*pi,0.0517317178751*pi) q[8];
U1q(0.554864977836986*pi,0.7378415016659599*pi) q[9];
U1q(0.424153593572138*pi,1.4272711495442003*pi) q[10];
U1q(0.436391555411659*pi,0.15807652728429922*pi) q[11];
U1q(0.107713004531732*pi,1.7243453341686*pi) q[12];
U1q(0.651245370169299*pi,1.0776598601479996*pi) q[13];
U1q(0.8608645397657*pi,1.5362006383909108*pi) q[14];
U1q(0.491231966305568*pi,0.46618101620230057*pi) q[15];
U1q(0.428169343657911*pi,0.2974807843499896*pi) q[16];
U1q(0.294540947960531*pi,1.3375350197557996*pi) q[17];
U1q(0.723009343890857*pi,1.1350887185844005*pi) q[18];
U1q(0.282536889485144*pi,0.13388686520210058*pi) q[19];
U1q(0.634832546891564*pi,0.7908325246074295*pi) q[20];
U1q(0.896740641478674*pi,0.7312447209998005*pi) q[21];
U1q(0.335491429717672*pi,0.9973587688891001*pi) q[22];
U1q(0.556907016728036*pi,0.07786837576430017*pi) q[23];
U1q(0.154795746777272*pi,0.5279406549768009*pi) q[24];
U1q(0.515726059327272*pi,1.3235460485978994*pi) q[25];
U1q(0.364916586707719*pi,0.35097067414320016*pi) q[26];
U1q(0.506450026845956*pi,0.28948827235189967*pi) q[27];
U1q(0.703859896564808*pi,1.4990154723722*pi) q[28];
U1q(0.350208832664839*pi,1.3731902575107995*pi) q[29];
U1q(0.268688319393054*pi,0.35243164599600973*pi) q[30];
U1q(0.943169043437359*pi,0.7823391733101399*pi) q[31];
U1q(0.36035388965339*pi,1.2685714257652005*pi) q[32];
U1q(0.855630436795485*pi,0.47127672285300015*pi) q[33];
U1q(0.537803197120211*pi,0.45681685166620056*pi) q[34];
U1q(0.234290717966165*pi,0.33699763468978006*pi) q[35];
U1q(0.147829184143346*pi,1.5804538851327*pi) q[36];
U1q(0.746071254775578*pi,0.6396886058318003*pi) q[37];
U1q(0.261590153664331*pi,0.32253849285950054*pi) q[38];
U1q(0.223823059334524*pi,0.7558076842952008*pi) q[39];
U1q(0.449406812764433*pi,1.4237099857106994*pi) q[40];
U1q(0.362903708955548*pi,0.9245732731395009*pi) q[41];
U1q(0.657241102749462*pi,1.1295550322888008*pi) q[42];
U1q(0.423712948007122*pi,1.9378960366419005*pi) q[43];
U1q(0.817391921535161*pi,1.8007397010972*pi) q[44];
U1q(0.547218328355591*pi,1.5713829700783002*pi) q[45];
U1q(0.154508598808634*pi,0.00818026520719961*pi) q[46];
U1q(0.567005728505194*pi,1.8787800215539008*pi) q[47];
U1q(0.273221941015982*pi,1.8611864237710005*pi) q[48];
U1q(0.259912102405587*pi,0.7171053570371999*pi) q[49];
U1q(0.727315587306633*pi,1.9070573865056009*pi) q[50];
U1q(0.659761572684621*pi,1.3036108375653601*pi) q[51];
U1q(0.452398387312337*pi,0.5666079319115997*pi) q[52];
U1q(0.635631346133392*pi,0.8552030395219994*pi) q[53];
U1q(0.224812792597366*pi,1.7098586369017799*pi) q[54];
U1q(0.768177895117572*pi,0.8431557631099*pi) q[55];
RZZ(0.5*pi) q[46],q[0];
RZZ(0.5*pi) q[1],q[22];
RZZ(0.5*pi) q[2],q[30];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[4],q[10];
RZZ(0.5*pi) q[5],q[42];
RZZ(0.5*pi) q[6],q[29];
RZZ(0.5*pi) q[23],q[7];
RZZ(0.5*pi) q[8],q[54];
RZZ(0.5*pi) q[9],q[41];
RZZ(0.5*pi) q[11],q[55];
RZZ(0.5*pi) q[15],q[12];
RZZ(0.5*pi) q[32],q[14];
RZZ(0.5*pi) q[35],q[16];
RZZ(0.5*pi) q[19],q[17];
RZZ(0.5*pi) q[51],q[18];
RZZ(0.5*pi) q[45],q[20];
RZZ(0.5*pi) q[21],q[28];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[37],q[25];
RZZ(0.5*pi) q[26],q[38];
RZZ(0.5*pi) q[27],q[40];
RZZ(0.5*pi) q[31],q[39];
RZZ(0.5*pi) q[33],q[47];
RZZ(0.5*pi) q[44],q[34];
RZZ(0.5*pi) q[53],q[43];
RZZ(0.5*pi) q[48],q[49];
RZZ(0.5*pi) q[50],q[52];
U1q(0.104756527286458*pi,1.6638304246778013*pi) q[0];
U1q(0.421369811384654*pi,0.7140589169601004*pi) q[1];
U1q(0.664476752690237*pi,0.4320702439584698*pi) q[2];
U1q(0.4332167010266*pi,0.3652767127281997*pi) q[3];
U1q(0.537158262084073*pi,1.3715619426494001*pi) q[4];
U1q(0.55354653870895*pi,1.0064759360076003*pi) q[5];
U1q(0.520920654545781*pi,0.3733269177856009*pi) q[6];
U1q(0.260311203017619*pi,0.5908760091916001*pi) q[7];
U1q(0.502452127654982*pi,0.8184221654640993*pi) q[8];
U1q(0.251424711655827*pi,1.8955225096944002*pi) q[9];
U1q(0.869012943309912*pi,0.10082758738949948*pi) q[10];
U1q(0.722409623077573*pi,1.5284052308803986*pi) q[11];
U1q(0.45202969171858*pi,1.075922999408899*pi) q[12];
U1q(0.796228592051292*pi,1.8274837361757008*pi) q[13];
U1q(0.889988985566234*pi,0.6920847484297994*pi) q[14];
U1q(0.667034360143648*pi,0.1521477575911998*pi) q[15];
U1q(0.305080647817065*pi,0.26447562125430046*pi) q[16];
U1q(0.758784886783928*pi,0.9384775619323005*pi) q[17];
U1q(0.589711992669038*pi,0.8748319404705001*pi) q[18];
U1q(0.57390195090302*pi,0.39061276058760086*pi) q[19];
U1q(0.381526695491422*pi,0.38897293724680004*pi) q[20];
U1q(0.510356979595606*pi,0.7621554229649998*pi) q[21];
U1q(0.858442861297743*pi,1.3899317927220984*pi) q[22];
U1q(0.74809409983568*pi,0.31038753174929923*pi) q[23];
U1q(0.773454258940245*pi,1.2224457296345008*pi) q[24];
U1q(0.685329610725672*pi,1.7990536359134008*pi) q[25];
U1q(0.65723421737767*pi,0.44896291677500066*pi) q[26];
U1q(0.604019691436705*pi,1.2739468047120006*pi) q[27];
U1q(0.10508268538626*pi,1.2098043924793984*pi) q[28];
U1q(0.513069648533321*pi,0.5384114351554992*pi) q[29];
U1q(0.579882163301345*pi,1.3878950033133997*pi) q[30];
U1q(0.714297333634202*pi,1.6094175967515092*pi) q[31];
U1q(0.187595273075912*pi,0.08626814406309968*pi) q[32];
U1q(0.737985899508811*pi,0.9882849585901994*pi) q[33];
U1q(0.306827898799899*pi,1.7488177520261985*pi) q[34];
U1q(0.198468743141243*pi,0.3875182273083997*pi) q[35];
U1q(0.871673903554528*pi,0.02362197855820014*pi) q[36];
U1q(0.39341217252438*pi,0.05103902911790037*pi) q[37];
U1q(0.235794907470275*pi,1.2194570949221006*pi) q[38];
U1q(0.0914987923708502*pi,1.9009534317123986*pi) q[39];
U1q(0.334948348031085*pi,1.7753746029389*pi) q[40];
U1q(0.682609322400046*pi,1.6495417255303018*pi) q[41];
U1q(0.674262626024359*pi,1.7896528938234013*pi) q[42];
U1q(0.60262213945566*pi,1.1263758696261004*pi) q[43];
U1q(0.27751257391793*pi,1.7633996882131004*pi) q[44];
U1q(0.569946522025651*pi,1.3036245543530995*pi) q[45];
U1q(0.625301016151516*pi,0.4141381007387004*pi) q[46];
U1q(0.792923955322286*pi,1.7657975589024009*pi) q[47];
U1q(0.328960167533878*pi,0.9269092828394996*pi) q[48];
U1q(0.343400903645402*pi,1.1731016588000003*pi) q[49];
U1q(0.683865580802465*pi,1.8644405153952999*pi) q[50];
U1q(0.692737049086765*pi,1.7633451986799002*pi) q[51];
U1q(0.266874290792449*pi,1.8864832954631012*pi) q[52];
U1q(0.462387589871305*pi,0.39770009702440134*pi) q[53];
U1q(0.761979693739913*pi,0.55907864487728*pi) q[54];
U1q(0.443667547237238*pi,1.2454543425621*pi) q[55];
RZZ(0.5*pi) q[0],q[38];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[2],q[36];
RZZ(0.5*pi) q[3],q[28];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[5],q[24];
RZZ(0.5*pi) q[23],q[6];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[8],q[46];
RZZ(0.5*pi) q[9],q[40];
RZZ(0.5*pi) q[33],q[10];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[12],q[49];
RZZ(0.5*pi) q[31],q[13];
RZZ(0.5*pi) q[14],q[34];
RZZ(0.5*pi) q[16],q[43];
RZZ(0.5*pi) q[17],q[51];
RZZ(0.5*pi) q[18],q[53];
RZZ(0.5*pi) q[44],q[20];
RZZ(0.5*pi) q[21],q[41];
RZZ(0.5*pi) q[22],q[27];
RZZ(0.5*pi) q[50],q[25];
RZZ(0.5*pi) q[26],q[45];
RZZ(0.5*pi) q[35],q[29];
RZZ(0.5*pi) q[48],q[30];
RZZ(0.5*pi) q[32],q[47];
RZZ(0.5*pi) q[52],q[42];
RZZ(0.5*pi) q[54],q[55];
U1q(0.388033638793209*pi,1.5758894870423*pi) q[0];
U1q(0.684172516893162*pi,0.6462975977492995*pi) q[1];
U1q(0.554508121619767*pi,1.5982684755523007*pi) q[2];
U1q(0.32692460822924*pi,1.3814931399145998*pi) q[3];
U1q(0.555570861431236*pi,0.09406544316490084*pi) q[4];
U1q(0.497409998587946*pi,1.2448527991895002*pi) q[5];
U1q(0.414939104568554*pi,0.2753880774425994*pi) q[6];
U1q(0.657873256388687*pi,0.7955732418060002*pi) q[7];
U1q(0.802541007464211*pi,1.489332408451201*pi) q[8];
U1q(0.341211023262018*pi,0.7584544021491002*pi) q[9];
U1q(0.492401787138884*pi,0.10203740989100041*pi) q[10];
U1q(0.707493694457468*pi,1.9312552320983016*pi) q[11];
U1q(0.426492860037185*pi,0.7396626833143003*pi) q[12];
U1q(0.445118618288311*pi,0.6099247052492984*pi) q[13];
U1q(0.404072328955207*pi,0.9665069675951994*pi) q[14];
U1q(0.68442866931747*pi,1.2887820305350992*pi) q[15];
U1q(0.618504931150035*pi,1.0541055536127004*pi) q[16];
U1q(0.539622949527658*pi,1.0814833575178007*pi) q[17];
U1q(0.540902068743147*pi,1.8577475045442*pi) q[18];
U1q(0.766603446737354*pi,1.1935966134898006*pi) q[19];
U1q(0.639506271702482*pi,0.7814654297534993*pi) q[20];
U1q(0.53537098825049*pi,0.9979785333759992*pi) q[21];
U1q(0.8667833429766*pi,0.5465666022619011*pi) q[22];
U1q(0.746567483528741*pi,0.5144700646018983*pi) q[23];
U1q(0.346614383739755*pi,0.6230506880622002*pi) q[24];
U1q(0.192186807743481*pi,0.2510863176094986*pi) q[25];
U1q(0.628054323992013*pi,1.2038336065572004*pi) q[26];
U1q(0.387617089432806*pi,0.03536512030269989*pi) q[27];
U1q(0.264006382371993*pi,1.9562118104914994*pi) q[28];
U1q(0.762158426867054*pi,1.8821184562831998*pi) q[29];
U1q(0.611191918690844*pi,0.15020358795639943*pi) q[30];
U1q(0.0833517848096564*pi,0.5772676511396*pi) q[31];
U1q(0.0412666903902544*pi,0.18473073716549848*pi) q[32];
U1q(0.668679250466509*pi,0.16707600701359837*pi) q[33];
U1q(0.379241875401415*pi,1.276901735921399*pi) q[34];
U1q(0.982239563805963*pi,0.11625305216390025*pi) q[35];
U1q(0.505715812548202*pi,1.7860950252325*pi) q[36];
U1q(0.443601345897564*pi,0.22842767862960045*pi) q[37];
U1q(0.843009363945102*pi,0.7329761460176982*pi) q[38];
U1q(0.253514599917359*pi,0.35071423375099897*pi) q[39];
U1q(0.701697651976901*pi,1.195297259933099*pi) q[40];
U1q(0.732056312224989*pi,1.0463120915780983*pi) q[41];
U1q(0.875409896117414*pi,1.4554071207322998*pi) q[42];
U1q(0.909360830736792*pi,0.7432956415260001*pi) q[43];
U1q(0.870721167062683*pi,1.6271224841555991*pi) q[44];
U1q(0.680804039772027*pi,0.47779135826660024*pi) q[45];
U1q(0.775260980489482*pi,0.28951591885230066*pi) q[46];
U1q(0.643376477649652*pi,1.1962146696576*pi) q[47];
U1q(0.741120035301902*pi,1.690759347422599*pi) q[48];
U1q(0.523196263468845*pi,1.9952232050360017*pi) q[49];
U1q(0.590116551946089*pi,1.4549562635545996*pi) q[50];
U1q(0.629530389384841*pi,0.5696136346103007*pi) q[51];
U1q(0.7311199179858*pi,1.6541557198512997*pi) q[52];
U1q(0.691290052739588*pi,0.22785902205340136*pi) q[53];
U1q(0.508260045089056*pi,1.3850103749918006*pi) q[54];
U1q(0.289896043620458*pi,0.26457736206669935*pi) q[55];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[1],q[3];
RZZ(0.5*pi) q[2],q[48];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[44],q[6];
RZZ(0.5*pi) q[7],q[53];
RZZ(0.5*pi) q[8],q[27];
RZZ(0.5*pi) q[9],q[29];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[11],q[49];
RZZ(0.5*pi) q[12],q[40];
RZZ(0.5*pi) q[13],q[45];
RZZ(0.5*pi) q[14],q[26];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[37],q[18];
RZZ(0.5*pi) q[21],q[31];
RZZ(0.5*pi) q[54],q[22];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[28],q[39];
RZZ(0.5*pi) q[33],q[30];
RZZ(0.5*pi) q[32],q[42];
RZZ(0.5*pi) q[35],q[38];
RZZ(0.5*pi) q[47],q[41];
RZZ(0.5*pi) q[50],q[43];
RZZ(0.5*pi) q[46],q[55];
RZZ(0.5*pi) q[52],q[51];
U1q(0.514106275253697*pi,0.8654242518997002*pi) q[0];
U1q(0.1220142424312*pi,0.5806051230083007*pi) q[1];
U1q(0.688380989187776*pi,1.6895502309048993*pi) q[2];
U1q(0.541159346013829*pi,1.4887058778388003*pi) q[3];
U1q(0.772293021711071*pi,0.22922832899860168*pi) q[4];
U1q(0.896798824017103*pi,0.49708552054320165*pi) q[5];
U1q(0.763305961459475*pi,1.2537845537666001*pi) q[6];
U1q(0.385497874616426*pi,1.723241503064699*pi) q[7];
U1q(0.467218120498096*pi,0.416227778974001*pi) q[8];
U1q(0.620922740467533*pi,1.1078199635023012*pi) q[9];
U1q(0.424728356200264*pi,0.6734969769877992*pi) q[10];
U1q(0.299255429837723*pi,0.1738898061859011*pi) q[11];
U1q(0.581695941206858*pi,1.7544775543952014*pi) q[12];
U1q(0.146785159065151*pi,0.09561341206800122*pi) q[13];
U1q(0.498487598439772*pi,1.8240470981290997*pi) q[14];
U1q(0.516147655266841*pi,1.1842976970396997*pi) q[15];
U1q(0.130631978761467*pi,0.22340346523879973*pi) q[16];
U1q(0.490932931002321*pi,0.7574884405122013*pi) q[17];
U1q(0.702488532507896*pi,1.491482013506399*pi) q[18];
U1q(0.446521504460967*pi,1.1398180858271*pi) q[19];
U1q(0.443411983556136*pi,1.8559472133381991*pi) q[20];
U1q(0.160459097664757*pi,0.2728892086395014*pi) q[21];
U1q(0.455115689295437*pi,1.7646005601066008*pi) q[22];
U1q(0.651661853746781*pi,0.3763553227799008*pi) q[23];
U1q(0.0996324539843513*pi,1.1928960203814007*pi) q[24];
U1q(0.697725566884569*pi,1.1471098543295994*pi) q[25];
U1q(0.299575101395978*pi,1.3177335772587*pi) q[26];
U1q(0.333753999206925*pi,0.0031163896515007394*pi) q[27];
U1q(0.27062358658508*pi,1.6432007758196008*pi) q[28];
U1q(0.586636773247911*pi,1.7242362835321998*pi) q[29];
U1q(0.243052331869504*pi,0.16437751401110035*pi) q[30];
U1q(0.109321517051886*pi,1.4303776056935007*pi) q[31];
U1q(0.421502990548531*pi,0.6766538283033015*pi) q[32];
U1q(0.360351745757643*pi,0.8868200908717014*pi) q[33];
U1q(0.440377825783085*pi,1.6907268153535995*pi) q[34];
U1q(0.42847335079536*pi,1.0970889369560002*pi) q[35];
U1q(0.303647384038558*pi,0.4037036090210009*pi) q[36];
U1q(0.369838165891765*pi,1.4450408370449992*pi) q[37];
U1q(0.664539489596261*pi,0.2510362691764989*pi) q[38];
U1q(0.274218566379447*pi,0.1272412623429986*pi) q[39];
U1q(0.601784717263909*pi,0.5491729520744997*pi) q[40];
U1q(0.183458427294124*pi,0.3091732906571991*pi) q[41];
U1q(0.491407352836082*pi,0.8389746643243008*pi) q[42];
U1q(0.638878885522816*pi,1.2999490735559007*pi) q[43];
U1q(0.356166274209471*pi,1.7929792170965015*pi) q[44];
U1q(0.501069015545041*pi,0.8504805927228993*pi) q[45];
U1q(0.596073498486852*pi,0.11018762496410162*pi) q[46];
U1q(0.488165064478005*pi,1.3025063143348987*pi) q[47];
U1q(0.658071715139777*pi,1.308500303916901*pi) q[48];
U1q(0.683562503833324*pi,1.0993446303159011*pi) q[49];
U1q(0.739500000788219*pi,0.6572542898897993*pi) q[50];
U1q(0.596167238506342*pi,0.8518123310659007*pi) q[51];
U1q(0.580491854900487*pi,0.03033478947480006*pi) q[52];
U1q(0.659270613654955*pi,1.7923791322670013*pi) q[53];
U1q(0.256094316529889*pi,0.2925195914897003*pi) q[54];
U1q(0.268755604998645*pi,1.6259747096079984*pi) q[55];
rz(2.9920665584599*pi) q[0];
rz(3.9671375301401*pi) q[1];
rz(2.6264959336975995*pi) q[2];
rz(2.985188131562399*pi) q[3];
rz(0.29686826363530017*pi) q[4];
rz(1.878183572875301*pi) q[5];
rz(1.0201641386433984*pi) q[6];
rz(1.0857729380192005*pi) q[7];
rz(2.3402929348225*pi) q[8];
rz(0.1773261410364988*pi) q[9];
rz(1.4077900092962992*pi) q[10];
rz(0.9946512745033012*pi) q[11];
rz(3.1733327076115003*pi) q[12];
rz(3.530774466086001*pi) q[13];
rz(2.314749440534701*pi) q[14];
rz(2.2613704842977*pi) q[15];
rz(0.9604460964638015*pi) q[16];
rz(1.057607323810199*pi) q[17];
rz(2.3160802231923014*pi) q[18];
rz(3.5987715625091994*pi) q[19];
rz(2.8263006519628*pi) q[20];
rz(2.874449943134799*pi) q[21];
rz(2.7160548761494994*pi) q[22];
rz(2.3428481312794993*pi) q[23];
rz(0.37936388154680145*pi) q[24];
rz(1.8709424163547013*pi) q[25];
rz(0.8874560612770992*pi) q[26];
rz(1.0083333558910006*pi) q[27];
rz(3.642123702316699*pi) q[28];
rz(3.6425165034101994*pi) q[29];
rz(1.8694259474005008*pi) q[30];
rz(1.8726615449294002*pi) q[31];
rz(3.8248740754366004*pi) q[32];
rz(1.0505814203241997*pi) q[33];
rz(1.6291983940195003*pi) q[34];
rz(3.2405825923619*pi) q[35];
rz(1.3935102097870988*pi) q[36];
rz(1.5063379780816*pi) q[37];
rz(1.6565870865886012*pi) q[38];
rz(1.3204732560819004*pi) q[39];
rz(0.5844314911289992*pi) q[40];
rz(3.5752644450387017*pi) q[41];
rz(2.3784811899021996*pi) q[42];
rz(1.8485642822871*pi) q[43];
rz(1.2575191776110017*pi) q[44];
rz(3.450691759929299*pi) q[45];
rz(1.1026577968025002*pi) q[46];
rz(2.686675278309199*pi) q[47];
rz(0.9678635249348986*pi) q[48];
rz(3.9822079082103983*pi) q[49];
rz(3.5443340213861987*pi) q[50];
rz(0.7234779813360994*pi) q[51];
rz(0.5085515722460983*pi) q[52];
rz(0.8679278999166016*pi) q[53];
rz(1.4331950176692008*pi) q[54];
rz(1.4622057586001986*pi) q[55];
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
measure q[40] -> c[40];
measure q[41] -> c[41];
measure q[42] -> c[42];
measure q[43] -> c[43];
measure q[44] -> c[44];
measure q[45] -> c[45];
measure q[46] -> c[46];
measure q[47] -> c[47];
measure q[48] -> c[48];
measure q[49] -> c[49];
measure q[50] -> c[50];
measure q[51] -> c[51];
measure q[52] -> c[52];
measure q[53] -> c[53];
measure q[54] -> c[54];
measure q[55] -> c[55];