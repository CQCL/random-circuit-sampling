OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.432790799593134*pi,1.03633593029391*pi) q[0];
U1q(0.207396949712615*pi,1.9196582297121954*pi) q[1];
U1q(1.22449490319879*pi,0.48119990370095944*pi) q[2];
U1q(3.133006384195696*pi,1.1704785245781308*pi) q[3];
U1q(0.451727569745141*pi,1.780061004113487*pi) q[4];
U1q(1.95679259355947*pi,0.6479148560501056*pi) q[5];
U1q(1.192762765078*pi,0.21079809149026416*pi) q[6];
U1q(0.308160618397719*pi,0.9644198340930199*pi) q[7];
U1q(1.68281006554337*pi,1.428859380317078*pi) q[8];
U1q(0.0342935003669837*pi,1.1839746442845271*pi) q[9];
U1q(0.171538184733897*pi,1.28169485270567*pi) q[10];
U1q(1.82278265139275*pi,1.7648846095083315*pi) q[11];
U1q(1.63431918618286*pi,1.6759194481161146*pi) q[12];
U1q(1.08386789407577*pi,1.6339450587807753*pi) q[13];
U1q(0.381732318533313*pi,0.9157318849456599*pi) q[14];
U1q(1.57697839425802*pi,1.7584862652618711*pi) q[15];
U1q(1.46168081109159*pi,0.5083908605894059*pi) q[16];
U1q(1.24278331350975*pi,1.5389381896566783*pi) q[17];
U1q(0.500187456101963*pi,0.414756673243321*pi) q[18];
U1q(0.352686456726687*pi,1.8626511824761791*pi) q[19];
U1q(1.69666065616025*pi,0.7213440026859702*pi) q[20];
U1q(0.643196182648118*pi,1.878540969111485*pi) q[21];
U1q(0.161574925866023*pi,1.29083550662467*pi) q[22];
U1q(0.547419279331907*pi,1.801487685722348*pi) q[23];
U1q(0.747364779915524*pi,1.704208380605357*pi) q[24];
U1q(0.308750903209454*pi,0.495298384363013*pi) q[25];
U1q(1.52749569437946*pi,1.8300266497420044*pi) q[26];
U1q(1.89553295359737*pi,0.5120144234927447*pi) q[27];
U1q(1.58122454960995*pi,1.0785558688550738*pi) q[28];
U1q(1.86009303445891*pi,1.9407818162009087*pi) q[29];
U1q(1.24017571696094*pi,0.606697993931823*pi) q[30];
U1q(1.68310802591015*pi,0.2775743556789587*pi) q[31];
U1q(0.894036868600483*pi,1.852222004672248*pi) q[32];
U1q(1.19741966826888*pi,0.5577636628608276*pi) q[33];
U1q(0.456923995103911*pi,1.4474281837494871*pi) q[34];
U1q(0.417683560891802*pi,1.9336233861065457*pi) q[35];
U1q(1.8153208615665*pi,0.11767407732911313*pi) q[36];
U1q(1.33147255242449*pi,0.24472068130936528*pi) q[37];
U1q(0.562221427247504*pi,1.537926884316946*pi) q[38];
U1q(1.51268411203963*pi,1.5979540700567663*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[18],q[1];
RZZ(0.5*pi) q[30],q[2];
RZZ(0.5*pi) q[3],q[29];
RZZ(0.5*pi) q[22],q[4];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[6],q[12];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[8],q[37];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[32],q[11];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[14],q[17];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[39],q[16];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[34],q[27];
RZZ(0.5*pi) q[36],q[38];
U1q(0.76561551412654*pi,1.0578751371884079*pi) q[0];
U1q(0.564802135363164*pi,1.203977001613888*pi) q[1];
U1q(0.362416126785228*pi,1.1534213068559493*pi) q[2];
U1q(0.807661903047913*pi,1.669550685936331*pi) q[3];
U1q(0.856888064499472*pi,1.0499755577780299*pi) q[4];
U1q(0.425665928725546*pi,1.0780108507474258*pi) q[5];
U1q(0.657991801068209*pi,1.157565855222534*pi) q[6];
U1q(0.555589962384919*pi,1.4997499948120399*pi) q[7];
U1q(0.288927762689252*pi,1.092867892899397*pi) q[8];
U1q(0.693216668995234*pi,0.7514958902511899*pi) q[9];
U1q(0.535029738080524*pi,1.1760893885721102*pi) q[10];
U1q(0.866192672261914*pi,1.2836995150112345*pi) q[11];
U1q(0.749235350508709*pi,0.8738587346339148*pi) q[12];
U1q(0.565327274671813*pi,0.6481373414761551*pi) q[13];
U1q(0.591556546991362*pi,1.1962182150609202*pi) q[14];
U1q(0.439467644233876*pi,1.6735099193783904*pi) q[15];
U1q(0.204361757880767*pi,1.5657539474189157*pi) q[16];
U1q(0.342635733504478*pi,0.2850525554213581*pi) q[17];
U1q(0.522554296948699*pi,1.3129171681463*pi) q[18];
U1q(0.730798959513361*pi,0.26434024583290006*pi) q[19];
U1q(0.594043874031748*pi,0.9443279003988505*pi) q[20];
U1q(0.774525472652871*pi,0.7141635495208201*pi) q[21];
U1q(0.451589040848999*pi,0.046785992414919875*pi) q[22];
U1q(0.202567585398612*pi,0.7779297791708601*pi) q[23];
U1q(0.472065986641445*pi,0.64035881978264*pi) q[24];
U1q(0.273927948157983*pi,0.4587032362660701*pi) q[25];
U1q(0.763537789777832*pi,0.28440155904089437*pi) q[26];
U1q(0.527385584942478*pi,0.6299312573956546*pi) q[27];
U1q(0.884846096388316*pi,0.0515329715805839*pi) q[28];
U1q(0.542679589023294*pi,1.1746758568581388*pi) q[29];
U1q(0.63374795036128*pi,0.4113895476458831*pi) q[30];
U1q(0.831465909478134*pi,0.5579552994925585*pi) q[31];
U1q(0.605692838069467*pi,1.0111853174309902*pi) q[32];
U1q(0.341346470839972*pi,0.6460054514973075*pi) q[33];
U1q(0.401481549749704*pi,1.3539518949704*pi) q[34];
U1q(0.670767405353141*pi,1.331104319489619*pi) q[35];
U1q(0.611420183377046*pi,1.2790966285833223*pi) q[36];
U1q(0.0767853223330855*pi,0.6994672389048653*pi) q[37];
U1q(0.941587105092049*pi,0.84735996279817*pi) q[38];
U1q(0.713840726168223*pi,0.20196744340541617*pi) q[39];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[3],q[32];
RZZ(0.5*pi) q[36],q[4];
RZZ(0.5*pi) q[27],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[33],q[7];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[10],q[18];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[14],q[30];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[19],q[24];
RZZ(0.5*pi) q[20],q[31];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[39],q[28];
RZZ(0.5*pi) q[34],q[37];
U1q(0.731884856680808*pi,0.19010425187159008*pi) q[0];
U1q(0.600409699693876*pi,0.44913391491148014*pi) q[1];
U1q(0.11707041143864*pi,0.07734895437006983*pi) q[2];
U1q(0.435242146744626*pi,1.8462408431411301*pi) q[3];
U1q(0.668671470049383*pi,1.8377591604115802*pi) q[4];
U1q(0.587294671648315*pi,0.6442510421965153*pi) q[5];
U1q(0.869425537470342*pi,0.3411781941291343*pi) q[6];
U1q(0.341403034519704*pi,1.71962337489978*pi) q[7];
U1q(0.23182127722329*pi,0.3914341951486979*pi) q[8];
U1q(0.408421885195291*pi,0.8816855516094799*pi) q[9];
U1q(0.209763609018393*pi,0.2136259642907099*pi) q[10];
U1q(0.661493818512976*pi,1.9927337400482816*pi) q[11];
U1q(0.768856075286236*pi,0.9036032013944748*pi) q[12];
U1q(0.717034715881337*pi,1.664998554376485*pi) q[13];
U1q(0.523590233151297*pi,0.07261269863667019*pi) q[14];
U1q(0.112447923739367*pi,1.8056381132329005*pi) q[15];
U1q(0.318411816594525*pi,0.6813084720332458*pi) q[16];
U1q(0.372267254881182*pi,1.1412847785521283*pi) q[17];
U1q(0.517776752226338*pi,0.19584837220784035*pi) q[18];
U1q(0.661224751609938*pi,1.7889072331300202*pi) q[19];
U1q(0.894460244115877*pi,0.8209151263744499*pi) q[20];
U1q(0.512557475272481*pi,1.4189389074660999*pi) q[21];
U1q(0.834364637562159*pi,0.9908473685021901*pi) q[22];
U1q(0.63881665864306*pi,1.2648774312101203*pi) q[23];
U1q(0.952706018073509*pi,0.0025808650963199398*pi) q[24];
U1q(0.804389895790092*pi,1.02806942003664*pi) q[25];
U1q(0.205147275540713*pi,0.08498949692145441*pi) q[26];
U1q(0.230614539476664*pi,0.4696972900783942*pi) q[27];
U1q(0.306627204855369*pi,1.8177520653320132*pi) q[28];
U1q(0.415332864607827*pi,1.1282972957446589*pi) q[29];
U1q(0.0149414251420191*pi,0.7359087455070927*pi) q[30];
U1q(0.253287267870795*pi,0.41248720526551885*pi) q[31];
U1q(0.440056077464382*pi,1.6318930617071503*pi) q[32];
U1q(0.397370915067579*pi,0.8434670685935175*pi) q[33];
U1q(0.387300927968656*pi,0.4145775596736998*pi) q[34];
U1q(0.214004930798572*pi,1.5018156498928001*pi) q[35];
U1q(0.51139972921827*pi,1.558815462254953*pi) q[36];
U1q(0.7737440936713*pi,1.5161765784820855*pi) q[37];
U1q(0.60208229467043*pi,0.0038081396258302647*pi) q[38];
U1q(0.183783174858603*pi,0.9493793407251259*pi) q[39];
RZZ(0.5*pi) q[3],q[0];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[19],q[7];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[10],q[25];
RZZ(0.5*pi) q[30],q[11];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[28],q[14];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[34],q[17];
RZZ(0.5*pi) q[31],q[21];
RZZ(0.5*pi) q[22],q[38];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[27],q[35];
RZZ(0.5*pi) q[29],q[32];
U1q(0.244511905236814*pi,1.9951362805706996*pi) q[0];
U1q(0.893743199077014*pi,1.7052936627611803*pi) q[1];
U1q(0.16787450409332*pi,1.6247914694355696*pi) q[2];
U1q(0.783794655273336*pi,1.0606894980616506*pi) q[3];
U1q(0.235802591069774*pi,0.07730339185085988*pi) q[4];
U1q(0.670363412736805*pi,0.07867395603563576*pi) q[5];
U1q(0.562877671183804*pi,1.569679917811344*pi) q[6];
U1q(0.176231499007047*pi,0.41280281754105985*pi) q[7];
U1q(0.364688448037905*pi,1.1428447223915672*pi) q[8];
U1q(0.359791620844348*pi,1.5236458516346705*pi) q[9];
U1q(0.72258200072677*pi,1.80788992793293*pi) q[10];
U1q(0.407262107633259*pi,0.49246202570362163*pi) q[11];
U1q(0.276538911513794*pi,0.2514663089099747*pi) q[12];
U1q(0.225072372264207*pi,0.15928340145408537*pi) q[13];
U1q(0.485027138904325*pi,0.8788469251453597*pi) q[14];
U1q(0.879265428693348*pi,0.43877153859742*pi) q[15];
U1q(0.445752896565742*pi,0.20245623360841591*pi) q[16];
U1q(0.660150768769789*pi,0.480023879007609*pi) q[17];
U1q(0.409191622931899*pi,1.5125032290340599*pi) q[18];
U1q(0.727715454489256*pi,1.6725381567028998*pi) q[19];
U1q(0.160809650797439*pi,1.5109793744417601*pi) q[20];
U1q(0.322541208050333*pi,1.3615327742695502*pi) q[21];
U1q(0.46412591551241*pi,1.3303259176033002*pi) q[22];
U1q(0.215040397794179*pi,1.4920925621272998*pi) q[23];
U1q(0.28907335891625*pi,0.7025889177992894*pi) q[24];
U1q(0.603848186175088*pi,1.99175172484797*pi) q[25];
U1q(0.705990345304831*pi,1.7732436319329352*pi) q[26];
U1q(0.560967228182521*pi,0.36518374841277446*pi) q[27];
U1q(0.76873599355938*pi,1.8725170684534538*pi) q[28];
U1q(0.67591946623577*pi,0.40954852068456926*pi) q[29];
U1q(0.408660069063576*pi,0.42246468235736323*pi) q[30];
U1q(0.628973913877332*pi,1.72020455176445*pi) q[31];
U1q(0.520163060349538*pi,1.6237346275240299*pi) q[32];
U1q(0.556370479650566*pi,0.4564261525564284*pi) q[33];
U1q(0.546968562794518*pi,1.1650264918276898*pi) q[34];
U1q(0.765596400291661*pi,0.4773730558838203*pi) q[35];
U1q(0.188977676449593*pi,0.9922034914710132*pi) q[36];
U1q(0.700615599886649*pi,0.6533827789536852*pi) q[37];
U1q(0.787586205436052*pi,1.9833764434814896*pi) q[38];
U1q(0.584385777141548*pi,0.47442876032639614*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[34],q[7];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[35],q[21];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[25],q[30];
RZZ(0.5*pi) q[36],q[28];
RZZ(0.5*pi) q[39],q[37];
U1q(0.390054589602657*pi,1.5122700499976993*pi) q[0];
U1q(0.597739378936054*pi,0.3447163535639497*pi) q[1];
U1q(0.494427721477759*pi,1.275817746617589*pi) q[2];
U1q(0.935642400843919*pi,0.8973826167968308*pi) q[3];
U1q(0.331264363373001*pi,0.5442038217850005*pi) q[4];
U1q(0.351245618754677*pi,0.6139796925972059*pi) q[5];
U1q(0.771559930669578*pi,0.519622369831664*pi) q[6];
U1q(0.552625454620742*pi,1.8966735992280306*pi) q[7];
U1q(0.345594648441561*pi,1.8848606122100282*pi) q[8];
U1q(0.587145203117477*pi,0.4958949593276003*pi) q[9];
U1q(0.499101090578623*pi,1.8907504990528992*pi) q[10];
U1q(0.683673262242085*pi,0.11226621541745097*pi) q[11];
U1q(0.484383739275491*pi,0.0864022292560751*pi) q[12];
U1q(0.415399069662454*pi,1.4454256266165242*pi) q[13];
U1q(0.234147656788379*pi,0.9463076150933301*pi) q[14];
U1q(0.70304137794031*pi,1.7563607374838703*pi) q[15];
U1q(0.106959927344063*pi,1.5661200110916056*pi) q[16];
U1q(0.644653206824829*pi,0.4905427725902882*pi) q[17];
U1q(0.334240397139581*pi,1.5466336282455995*pi) q[18];
U1q(0.642874629411536*pi,1.6263968954380292*pi) q[19];
U1q(0.478091582013197*pi,1.7288568196019511*pi) q[20];
U1q(0.0843898538512575*pi,0.06324930811124041*pi) q[21];
U1q(0.269967523990162*pi,0.4822273778682007*pi) q[22];
U1q(0.416530252887428*pi,1.8755166024113308*pi) q[23];
U1q(0.575489417554498*pi,1.0833724586748001*pi) q[24];
U1q(0.352247381576727*pi,0.65441309344458*pi) q[25];
U1q(0.769671938574895*pi,1.301591258628184*pi) q[26];
U1q(0.197591008272845*pi,1.8833884501322462*pi) q[27];
U1q(0.52883537069573*pi,0.36604394568774357*pi) q[28];
U1q(0.558004379986013*pi,1.0810255056041598*pi) q[29];
U1q(0.820813366530477*pi,1.2331245212856228*pi) q[30];
U1q(0.193988909512093*pi,1.7802989675201175*pi) q[31];
U1q(0.816869084905412*pi,0.27328743954194934*pi) q[32];
U1q(0.399433309615429*pi,1.924849843347367*pi) q[33];
U1q(0.156578909036033*pi,0.19302821054284003*pi) q[34];
U1q(0.407268132344077*pi,0.9608433327454993*pi) q[35];
U1q(0.701730372371733*pi,0.4834703898354027*pi) q[36];
U1q(0.465075390055145*pi,0.40532957966766503*pi) q[37];
U1q(0.796723387605168*pi,1.3930394026508992*pi) q[38];
U1q(0.736527637592027*pi,0.4193497440628464*pi) q[39];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[29],q[2];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[9],q[5];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[8],q[21];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[13],q[14];
RZZ(0.5*pi) q[16],q[28];
RZZ(0.5*pi) q[20],q[17];
RZZ(0.5*pi) q[18],q[38];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[25],q[22];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[37],q[30];
RZZ(0.5*pi) q[32],q[35];
RZZ(0.5*pi) q[34],q[36];
U1q(0.230210629459202*pi,0.7631644563291005*pi) q[0];
U1q(0.137941819926837*pi,1.6577650032395006*pi) q[1];
U1q(0.444066876519462*pi,1.96055680603496*pi) q[2];
U1q(0.368415027744646*pi,1.125471724989632*pi) q[3];
U1q(0.451462892308251*pi,0.4494070514619004*pi) q[4];
U1q(0.351302314406768*pi,1.5906380434792364*pi) q[5];
U1q(0.316264499438229*pi,0.774474171302165*pi) q[6];
U1q(0.248503219313746*pi,0.20673050218658062*pi) q[7];
U1q(0.368686665980428*pi,0.038760683285378406*pi) q[8];
U1q(0.306251949357186*pi,0.08298075024289986*pi) q[9];
U1q(0.411373036508931*pi,1.3981238077788998*pi) q[10];
U1q(0.285936832654031*pi,0.7199455532858128*pi) q[11];
U1q(0.780690875409996*pi,1.8699415647224154*pi) q[12];
U1q(0.78228532872784*pi,1.2573499251349745*pi) q[13];
U1q(0.904354490816887*pi,1.3855065680827803*pi) q[14];
U1q(0.34547438255802*pi,0.7034623234412702*pi) q[15];
U1q(0.903652296119441*pi,1.2308069116812064*pi) q[16];
U1q(0.0169820681329067*pi,1.7223440555576772*pi) q[17];
U1q(0.627320626641915*pi,1.8307164482838*pi) q[18];
U1q(0.510010211208536*pi,1.3510048784222999*pi) q[19];
U1q(0.566100026776386*pi,0.5389304359378713*pi) q[20];
U1q(0.729700172495667*pi,0.8844991759183003*pi) q[21];
U1q(0.624460490348811*pi,0.6696393025226008*pi) q[22];
U1q(0.503628395500235*pi,1.3187705661219002*pi) q[23];
U1q(0.690522550848434*pi,0.025519812234200856*pi) q[24];
U1q(0.272743650345132*pi,1.3586584335073297*pi) q[25];
U1q(0.505602494389731*pi,0.3468976922701241*pi) q[26];
U1q(0.0278414182021229*pi,0.23677977046884635*pi) q[27];
U1q(0.692642873006923*pi,0.9449540165536341*pi) q[28];
U1q(0.419858536009359*pi,1.9147033803114084*pi) q[29];
U1q(0.0340136022071425*pi,0.9680833614041227*pi) q[30];
U1q(0.50997516034832*pi,0.685229185162779*pi) q[31];
U1q(0.617002837933829*pi,1.0592821286934004*pi) q[32];
U1q(0.359900092852517*pi,0.9040514047321278*pi) q[33];
U1q(0.302434915197717*pi,0.4928939589244994*pi) q[34];
U1q(0.366341328448072*pi,1.5567594942233*pi) q[35];
U1q(0.574348462796911*pi,1.2069210770064327*pi) q[36];
U1q(0.59980404154903*pi,0.19834185508586621*pi) q[37];
U1q(0.27949021441519*pi,1.2200042390437993*pi) q[38];
U1q(0.659006410088456*pi,1.7903955515637158*pi) q[39];
RZZ(0.5*pi) q[0],q[30];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[24];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[6],q[31];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[18],q[8];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[13],q[35];
RZZ(0.5*pi) q[29],q[14];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[19],q[17];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[26],q[25];
RZZ(0.5*pi) q[28],q[38];
RZZ(0.5*pi) q[37],q[36];
U1q(0.198751570852002*pi,0.3227115196153001*pi) q[0];
U1q(0.15248288317644*pi,1.2384827780691996*pi) q[1];
U1q(0.249697131443958*pi,1.1290050278440589*pi) q[2];
U1q(0.630822715928708*pi,1.5739616507651313*pi) q[3];
U1q(0.337455938653035*pi,1.4726610349559017*pi) q[4];
U1q(0.304283537872081*pi,1.7649810610872052*pi) q[5];
U1q(0.466697932252305*pi,0.4393984262104649*pi) q[6];
U1q(0.484679443476949*pi,1.0989418712233991*pi) q[7];
U1q(0.795418345696631*pi,0.12948816937357854*pi) q[8];
U1q(0.516596375394121*pi,1.197948601757*pi) q[9];
U1q(0.77862902385515*pi,1.4248504558723987*pi) q[10];
U1q(0.788796228776687*pi,0.15432649812633237*pi) q[11];
U1q(0.522267008045751*pi,1.0204153781954144*pi) q[12];
U1q(0.247087574752203*pi,1.7068970132898755*pi) q[13];
U1q(0.760026624858057*pi,0.8782882365250995*pi) q[14];
U1q(0.831340190545469*pi,1.9387590976626718*pi) q[15];
U1q(0.558827049965266*pi,0.45756146585680746*pi) q[16];
U1q(0.170072071920728*pi,0.44531817440067734*pi) q[17];
U1q(0.591106258957369*pi,0.9411637891581996*pi) q[18];
U1q(0.652783681918697*pi,1.3711539418154999*pi) q[19];
U1q(0.470828925268161*pi,0.3220653898397714*pi) q[20];
U1q(0.692780391927133*pi,0.2471413407452001*pi) q[21];
U1q(0.727080283218262*pi,0.8839237429532005*pi) q[22];
U1q(0.584912240843238*pi,0.13478955931499925*pi) q[23];
U1q(0.246945221506153*pi,0.7129751489292993*pi) q[24];
U1q(0.862433237843836*pi,0.19093448478420072*pi) q[25];
U1q(0.114146293647371*pi,1.5979778869122043*pi) q[26];
U1q(0.596873443167594*pi,1.1284113373810456*pi) q[27];
U1q(0.783960515446083*pi,0.317008216316224*pi) q[28];
U1q(0.103325628145153*pi,1.5829982780005096*pi) q[29];
U1q(0.524241288552863*pi,0.6251111172721231*pi) q[30];
U1q(0.504944137723967*pi,0.13782549752505702*pi) q[31];
U1q(0.356310051323309*pi,0.8959156380966*pi) q[32];
U1q(0.697832248877073*pi,0.5532923233861275*pi) q[33];
U1q(0.325987708576984*pi,1.0111574160860002*pi) q[34];
U1q(0.617589604293216*pi,0.9698449792147006*pi) q[35];
U1q(0.333730826398173*pi,0.7098216841060729*pi) q[36];
U1q(0.358444981230606*pi,0.26295657809426487*pi) q[37];
U1q(0.0915257375793417*pi,1.5333252891065001*pi) q[38];
U1q(0.400587894228183*pi,0.2789621724942659*pi) q[39];
rz(2.7281160873131007*pi) q[0];
rz(3.598467364044099*pi) q[1];
rz(2.3783892585010413*pi) q[2];
rz(3.575098558934169*pi) q[3];
rz(0.6078565451546005*pi) q[4];
rz(3.2931972512161938*pi) q[5];
rz(0.8387008912268357*pi) q[6];
rz(2.7654745647552996*pi) q[7];
rz(2.880495969748022*pi) q[8];
rz(0.13877993926329957*pi) q[9];
rz(0.707744459702301*pi) q[10];
rz(1.0363903962173673*pi) q[11];
rz(3.6173549942527856*pi) q[12];
rz(0.21354994778392467*pi) q[13];
rz(0.27635442299530055*pi) q[14];
rz(0.5607562724178266*pi) q[15];
rz(3.7107979146696923*pi) q[16];
rz(0.46711721594641986*pi) q[17];
rz(2.2238460513457987*pi) q[18];
rz(0.1537579037912007*pi) q[19];
rz(2.37694488910633*pi) q[20];
rz(3.7283420329176007*pi) q[21];
rz(1.2779314767623013*pi) q[22];
rz(0.13637427921569945*pi) q[23];
rz(1.6975689378969*pi) q[24];
rz(3.1781462522853*pi) q[25];
rz(3.6210211918353963*pi) q[26];
rz(3.0241069070632545*pi) q[27];
rz(3.563413579322476*pi) q[28];
rz(1.6339872069601906*pi) q[29];
rz(1.4625553372575766*pi) q[30];
rz(3.0495952029350413*pi) q[31];
rz(0.9265404016899996*pi) q[32];
rz(0.7403130377406733*pi) q[33];
rz(2.0592803241806994*pi) q[34];
rz(0.6100689267021995*pi) q[35];
rz(1.7298648958093068*pi) q[36];
rz(0.2933753741454339*pi) q[37];
rz(1.0805180658254017*pi) q[38];
rz(3.865157349225834*pi) q[39];
U1q(0.198751570852002*pi,0.0508276069283922*pi) q[0];
U1q(0.15248288317644*pi,1.836950142113254*pi) q[1];
U1q(1.24969713144396*pi,0.507394286345163*pi) q[2];
U1q(1.63082271592871*pi,0.149060209699264*pi) q[3];
U1q(1.33745593865304*pi,1.080517580110488*pi) q[4];
U1q(0.304283537872081*pi,0.0581783123034211*pi) q[5];
U1q(1.4666979322523*pi,0.278099317437352*pi) q[6];
U1q(1.48467944347695*pi,0.864416435978734*pi) q[7];
U1q(3.795418345696631*pi,0.00998413912162555*pi) q[8];
U1q(1.51659637539412*pi,0.336728541020274*pi) q[9];
U1q(0.77862902385515*pi,1.1325949155746788*pi) q[10];
U1q(0.788796228776687*pi,0.19071689434367*pi) q[11];
U1q(0.522267008045751*pi,1.6377703724482031*pi) q[12];
U1q(0.247087574752203*pi,0.92044696107377*pi) q[13];
U1q(0.760026624858057*pi,0.15464265952039*pi) q[14];
U1q(0.831340190545469*pi,1.4995153700804549*pi) q[15];
U1q(0.558827049965266*pi,1.168359380526502*pi) q[16];
U1q(1.17007207192073*pi,1.9124353903470688*pi) q[17];
U1q(1.59110625895737*pi,0.165009840504001*pi) q[18];
U1q(3.652783681918698*pi,0.524911845606664*pi) q[19];
U1q(1.47082892526816*pi,1.699010278946081*pi) q[20];
U1q(0.692780391927133*pi,0.975483373662819*pi) q[21];
U1q(0.727080283218262*pi,1.161855219715459*pi) q[22];
U1q(1.58491224084324*pi,1.27116383853067*pi) q[23];
U1q(0.246945221506153*pi,1.410544086826167*pi) q[24];
U1q(0.862433237843836*pi,0.369080737069443*pi) q[25];
U1q(1.11414629364737*pi,0.2189990787476*pi) q[26];
U1q(1.59687344316759*pi,1.15251824444422*pi) q[27];
U1q(1.78396051544608*pi,0.880421795638704*pi) q[28];
U1q(1.10332562814515*pi,0.216985484960731*pi) q[29];
U1q(1.52424128855286*pi,1.087666454529702*pi) q[30];
U1q(0.504944137723967*pi,0.187420700460113*pi) q[31];
U1q(0.356310051323309*pi,0.822456039786529*pi) q[32];
U1q(0.697832248877073*pi,0.293605361126751*pi) q[33];
U1q(1.32598770857698*pi,0.0704377402667326*pi) q[34];
U1q(1.61758960429322*pi,0.579913905916918*pi) q[35];
U1q(1.33373082639817*pi,1.43968657991538*pi) q[36];
U1q(1.35844498123061*pi,1.556331952239617*pi) q[37];
U1q(0.0915257375793417*pi,1.613843354931896*pi) q[38];
U1q(1.40058789422818*pi,1.1441195217200169*pi) q[39];
RZZ(0.5*pi) q[0],q[30];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[24];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[6],q[31];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[18],q[8];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[13],q[35];
RZZ(0.5*pi) q[29],q[14];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[19],q[17];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[26],q[25];
RZZ(0.5*pi) q[28],q[38];
RZZ(0.5*pi) q[37],q[36];
U1q(1.2302106294592*pi,0.491280543642117*pi) q[0];
U1q(0.137941819926837*pi,0.25623236728359*pi) q[1];
U1q(1.44406687651946*pi,1.6758425081543344*pi) q[2];
U1q(3.631584972255354*pi,1.597550135474702*pi) q[3];
U1q(1.45146289230825*pi,0.10377156360456574*pi) q[4];
U1q(1.35130231440677*pi,1.883835294695458*pi) q[5];
U1q(1.31626449943823*pi,1.9430235723456681*pi) q[6];
U1q(3.751496780686254*pi,0.7566278050155986*pi) q[7];
U1q(3.631313334019572*pi,0.10071162520977173*pi) q[8];
U1q(1.30625194935719*pi,1.451696392534331*pi) q[9];
U1q(1.41137303650893*pi,1.1058682674811702*pi) q[10];
U1q(0.285936832654031*pi,0.7563359495031401*pi) q[11];
U1q(1.78069087541*pi,0.48729655897528*pi) q[12];
U1q(0.78228532872784*pi,0.47089987291886004*pi) q[13];
U1q(1.90435449081689*pi,0.6618609910780799*pi) q[14];
U1q(0.34547438255802*pi,0.2642185958590799*pi) q[15];
U1q(1.90365229611944*pi,0.94160482635095*pi) q[16];
U1q(1.01698206813291*pi,1.6354095091901106*pi) q[17];
U1q(3.372679373358086*pi,1.275457181378405*pi) q[18];
U1q(3.489989788791464*pi,0.5450609089998245*pi) q[19];
U1q(3.433899973223614*pi,0.4821452328479674*pi) q[20];
U1q(1.72970017249567*pi,0.612841208835926*pi) q[21];
U1q(1.62446049034881*pi,0.9475707792848702*pi) q[22];
U1q(3.496371604499765*pi,1.0871828317237717*pi) q[23];
U1q(0.690522550848434*pi,0.7230887501311498*pi) q[24];
U1q(0.272743650345132*pi,1.53680468579262*pi) q[25];
U1q(1.50560249438973*pi,1.470079273389683*pi) q[26];
U1q(3.972158581797878*pi,1.0441498113563281*pi) q[27];
U1q(3.307357126993078*pi,1.2524759954012965*pi) q[28];
U1q(3.580141463990641*pi,0.8852803826498685*pi) q[29];
U1q(3.965986397792856*pi,1.744694210397752*pi) q[30];
U1q(1.50997516034832*pi,0.734824388097821*pi) q[31];
U1q(0.617002837933829*pi,0.985822530383335*pi) q[32];
U1q(0.359900092852517*pi,0.644364442472789*pi) q[33];
U1q(3.697565084802283*pi,1.5887011974282679*pi) q[34];
U1q(3.633658671551928*pi,0.9929993909083499*pi) q[35];
U1q(1.57434846279691*pi,1.9425871870150253*pi) q[36];
U1q(3.4001959584509702*pi,1.620946675247989*pi) q[37];
U1q(1.27949021441519*pi,1.30052230486928*pi) q[38];
U1q(1.65900641008846*pi,1.6326861426505335*pi) q[39];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[29],q[2];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[9],q[5];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[8],q[21];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[13],q[14];
RZZ(0.5*pi) q[16],q[28];
RZZ(0.5*pi) q[20],q[17];
RZZ(0.5*pi) q[18],q[38];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[25],q[22];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[37],q[30];
RZZ(0.5*pi) q[32],q[35];
RZZ(0.5*pi) q[34],q[36];
U1q(3.609945410397343*pi,0.7421749499734684*pi) q[0];
U1q(1.59773937893605*pi,1.9431837176080302*pi) q[1];
U1q(0.494427721477759*pi,0.9911034487370012*pi) q[2];
U1q(3.064357599156081*pi,1.825639243667553*pi) q[3];
U1q(1.331264363373*pi,0.1985683339276676*pi) q[4];
U1q(3.648754381245323*pi,1.8604936455774852*pi) q[5];
U1q(1.77155993066958*pi,0.6881717708751542*pi) q[6];
U1q(3.4473745453792572*pi,1.0666847079741477*pi) q[7];
U1q(3.345594648441561*pi,0.2546116962851599*pi) q[8];
U1q(1.58714520311748*pi,1.8646106016189958*pi) q[9];
U1q(3.500898909421377*pi,0.6132415762071184*pi) q[10];
U1q(0.683673262242085*pi,0.14865661163478006*pi) q[11];
U1q(3.515616260724509*pi,0.2708358944416478*pi) q[12];
U1q(1.41539906966245*pi,0.65897557440043*pi) q[13];
U1q(1.23414765678838*pi,1.1010599440675253*pi) q[14];
U1q(0.70304137794031*pi,0.31711700990165004*pi) q[15];
U1q(3.893040072655936*pi,0.606291726940601*pi) q[16];
U1q(0.644653206824829*pi,0.4036082262227505*pi) q[17];
U1q(3.665759602860418*pi,0.5595400014166532*pi) q[18];
U1q(3.357125370588464*pi,0.2696688919841185*pi) q[19];
U1q(1.4780915820132*pi,0.29221884918391794*pi) q[20];
U1q(3.9156101461487425*pi,1.4340910766429888*pi) q[21];
U1q(1.26996752399016*pi,1.1349827039393223*pi) q[22];
U1q(1.41653025288743*pi,1.53043679543432*pi) q[23];
U1q(0.575489417554498*pi,0.7809413965717602*pi) q[24];
U1q(1.35224738157673*pi,0.832559345729863*pi) q[25];
U1q(0.769671938574895*pi,0.42477283974773294*pi) q[26];
U1q(3.802408991727155*pi,0.3975411316929982*pi) q[27];
U1q(1.52883537069573*pi,1.8313860662671928*pi) q[28];
U1q(3.441995620013987*pi,1.7189582573570805*pi) q[29];
U1q(3.179186633469523*pi,0.4796530505162422*pi) q[30];
U1q(3.806011090487907*pi,1.639754605740488*pi) q[31];
U1q(1.81686908490541*pi,0.199827841231928*pi) q[32];
U1q(1.39943330961543*pi,1.66516288108802*pi) q[33];
U1q(3.843421090963966*pi,1.888566945809884*pi) q[34];
U1q(3.592731867655923*pi,1.5889155523861458*pi) q[35];
U1q(0.701730372371733*pi,1.2191364998439993*pi) q[36];
U1q(3.534924609944855*pi,0.41395895066617805*pi) q[37];
U1q(3.203276612394831*pi,1.1274871412622174*pi) q[38];
U1q(1.73652763759203*pi,0.2616403351496668*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[34],q[7];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[35],q[21];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[25],q[30];
RZZ(0.5*pi) q[36],q[28];
RZZ(0.5*pi) q[39],q[37];
U1q(3.755488094763185*pi,0.2593087194004613*pi) q[0];
U1q(1.89374319907701*pi,1.5826064084108005*pi) q[1];
U1q(1.16787450409332*pi,1.3400771715549817*pi) q[2];
U1q(1.78379465527334*pi,1.662332362402711*pi) q[3];
U1q(1.23580259106977*pi,1.6654687638617665*pi) q[4];
U1q(1.6703634127368*pi,0.3957993821390575*pi) q[5];
U1q(3.437122328816196*pi,1.6381142228954673*pi) q[6];
U1q(3.823768500992952*pi,1.5505554896611233*pi) q[7];
U1q(1.3646884480379*pi,0.51259580646669*pi) q[8];
U1q(3.359791620844348*pi,0.8368597093119245*pi) q[9];
U1q(3.277417999273231*pi,1.6961021473270987*pi) q[10];
U1q(1.40726210763326*pi,1.52885242192096*pi) q[11];
U1q(1.27653891151379*pi,1.1057718147877482*pi) q[12];
U1q(3.774927627735793*pi,1.9451177995628717*pi) q[13];
U1q(0.485027138904325*pi,0.03359925411955578*pi) q[14];
U1q(1.87926542869335*pi,1.9995278110152404*pi) q[15];
U1q(1.44575289656574*pi,1.969955504423786*pi) q[16];
U1q(3.66015076876979*pi,0.3930893326400704*pi) q[17];
U1q(3.590808377068101*pi,0.5936704006281841*pi) q[18];
U1q(3.272284545510744*pi,1.2235276307192544*pi) q[19];
U1q(3.160809650797439*pi,0.07434140402371803*pi) q[20];
U1q(1.32254120805033*pi,0.13580761048467224*pi) q[21];
U1q(1.46412591551241*pi,1.9830812436745022*pi) q[22];
U1q(0.215040397794179*pi,1.1470127551502947*pi) q[23];
U1q(1.28907335891625*pi,0.4001578556962002*pi) q[24];
U1q(3.603848186175088*pi,0.4952207143264673*pi) q[25];
U1q(1.70599034530483*pi,0.8964252130524929*pi) q[26];
U1q(1.56096722818252*pi,0.91574583341245*pi) q[27];
U1q(1.76873599355938*pi,0.33785918903291073*pi) q[28];
U1q(3.67591946623577*pi,0.39043524227666904*pi) q[29];
U1q(1.40866006906358*pi,1.2903128894445102*pi) q[30];
U1q(1.62897391387733*pi,0.6998490214961501*pi) q[31];
U1q(1.52016306034954*pi,1.849380653249845*pi) q[32];
U1q(3.4436295203494343*pi,0.1335865718789564*pi) q[33];
U1q(1.54696856279452*pi,0.9165686645250363*pi) q[34];
U1q(3.234403599708339*pi,1.0723858292478279*pi) q[35];
U1q(1.18897767644959*pi,0.7278696014796053*pi) q[36];
U1q(3.299384400113351*pi,1.165905751380158*pi) q[37];
U1q(3.212413794563949*pi,1.5371501004316177*pi) q[38];
U1q(3.415614222858451*pi,0.2065613188861115*pi) q[39];
RZZ(0.5*pi) q[3],q[0];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[19],q[7];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[10],q[25];
RZZ(0.5*pi) q[30],q[11];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[28],q[14];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[34],q[17];
RZZ(0.5*pi) q[31],q[21];
RZZ(0.5*pi) q[22],q[38];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[27],q[35];
RZZ(0.5*pi) q[29],q[32];
U1q(3.268115143319192*pi,0.06434074809957124*pi) q[0];
U1q(1.60040969969388*pi,0.32644666056109983*pi) q[1];
U1q(3.11707041143864*pi,0.8875196866204877*pi) q[2];
U1q(0.435242146744626*pi,0.44788370748218087*pi) q[3];
U1q(0.668671470049383*pi,1.4259245324224867*pi) q[4];
U1q(0.587294671648315*pi,0.9613764682999357*pi) q[5];
U1q(3.130574462529658*pi,1.866615946577677*pi) q[6];
U1q(3.341403034519704*pi,0.24373493230240806*pi) q[7];
U1q(1.23182127722329*pi,0.26400633370955795*pi) q[8];
U1q(0.408421885195291*pi,0.19489940928673666*pi) q[9];
U1q(1.20976360901839*pi,0.2903661109693232*pi) q[10];
U1q(3.661493818512977*pi,0.02858070757630715*pi) q[11];
U1q(0.768856075286236*pi,1.7579087072722386*pi) q[12];
U1q(3.2829652841186627*pi,1.4394026466404721*pi) q[13];
U1q(0.523590233151297*pi,1.2273650276108548*pi) q[14];
U1q(3.887552076260633*pi,0.6326612363797661*pi) q[15];
U1q(1.31841181659453*pi,0.44880774284861014*pi) q[16];
U1q(3.627732745118818*pi,0.7318284330955486*pi) q[17];
U1q(1.51777675222634*pi,0.9103252574544092*pi) q[18];
U1q(1.66122475160994*pi,1.1071585542921278*pi) q[19];
U1q(3.105539755884123*pi,0.764405652091027*pi) q[20];
U1q(1.51255747527248*pi,1.1932137436812142*pi) q[21];
U1q(3.165635362437841*pi,0.3225597927756567*pi) q[22];
U1q(0.63881665864306*pi,0.9197976242331152*pi) q[23];
U1q(3.047293981926494*pi,0.10016590839916972*pi) q[24];
U1q(0.804389895790092*pi,1.5315384095151343*pi) q[25];
U1q(1.20514727554071*pi,0.584679348063976*pi) q[26];
U1q(0.230614539476664*pi,1.0202593750780702*pi) q[27];
U1q(1.30662720485537*pi,0.3926241921543574*pi) q[28];
U1q(0.415332864607827*pi,0.10918401733675287*pi) q[29];
U1q(1.01494142514202*pi,1.6037569525942397*pi) q[30];
U1q(1.2532872678708*pi,0.3921316749972159*pi) q[31];
U1q(1.44005607746438*pi,0.8575390874329671*pi) q[32];
U1q(1.39737091506758*pi,1.7465456558418708*pi) q[33];
U1q(1.38730092796866*pi,0.16611973237104172*pi) q[34];
U1q(3.785995069201427*pi,0.04794323523884381*pi) q[35];
U1q(3.51139972921827*pi,0.16125763069566235*pi) q[36];
U1q(3.7737440936713*pi,1.3031119518517613*pi) q[37];
U1q(3.39791770532957*pi,0.5167184042872774*pi) q[38];
U1q(3.183783174858603*pi,0.7316107384873807*pi) q[39];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[3],q[32];
RZZ(0.5*pi) q[36],q[4];
RZZ(0.5*pi) q[27],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[33],q[7];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[10],q[18];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[14],q[30];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[19],q[24];
RZZ(0.5*pi) q[20],q[31];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[39],q[28];
RZZ(0.5*pi) q[34],q[37];
U1q(3.23438448587346*pi,1.196569862782761*pi) q[0];
U1q(3.564802135363164*pi,0.5716035738586891*pi) q[1];
U1q(1.36241612678523*pi,1.9635920391063764*pi) q[2];
U1q(3.807661903047913*pi,0.2711935502773808*pi) q[3];
U1q(0.856888064499472*pi,1.6381409297889356*pi) q[4];
U1q(1.42566592872555*pi,0.3951362768508403*pi) q[5];
U1q(3.342008198931791*pi,1.0502282854842768*pi) q[6];
U1q(0.555589962384919*pi,1.023861552214667*pi) q[7];
U1q(0.288927762689252*pi,1.9654400314602576*pi) q[8];
U1q(3.693216668995235*pi,0.06470974792843576*pi) q[9];
U1q(0.535029738080524*pi,0.2528295352507328*pi) q[10];
U1q(1.86619267226191*pi,1.319546482539257*pi) q[11];
U1q(1.74923535050871*pi,1.7281642405116786*pi) q[12];
U1q(1.56532727467181*pi,1.4562638595407975*pi) q[13];
U1q(0.591556546991362*pi,0.35097054403511496*pi) q[14];
U1q(3.439467644233877*pi,1.7647894302342761*pi) q[15];
U1q(1.20436175788077*pi,0.564362267462932*pi) q[16];
U1q(1.34263573350448*pi,1.588060656226315*pi) q[17];
U1q(0.522554296948699*pi,1.0273940533928672*pi) q[18];
U1q(0.730798959513361*pi,1.582591566994998*pi) q[19];
U1q(3.405956125968252*pi,1.6409928780666267*pi) q[20];
U1q(1.77452547265287*pi,0.8979891016264889*pi) q[21];
U1q(1.451589040849*pi,1.2666211688629287*pi) q[22];
U1q(0.202567585398612*pi,1.4328499721938561*pi) q[23];
U1q(3.527934013358555*pi,1.4623879537128497*pi) q[24];
U1q(0.273927948157983*pi,1.9621722257445544*pi) q[25];
U1q(1.76353778977783*pi,0.7840914101834162*pi) q[26];
U1q(1.52738558494248*pi,0.1804933423953301*pi) q[27];
U1q(0.884846096388316*pi,1.6264050984029335*pi) q[28];
U1q(0.542679589023294*pi,1.1555625784502328*pi) q[29];
U1q(3.63374795036128*pi,1.9282761504554466*pi) q[30];
U1q(3.168534090521865*pi,0.24666358077016803*pi) q[31];
U1q(3.394307161930533*pi,0.47824683170913396*pi) q[32];
U1q(1.34134647083997*pi,0.5490840387456606*pi) q[33];
U1q(1.4014815497497*pi,1.2267453970743345*pi) q[34];
U1q(3.329232594646859*pi,1.2186545656420287*pi) q[35];
U1q(0.611420183377046*pi,0.8815387970240325*pi) q[36];
U1q(0.0767853223330855*pi,0.4864026122745413*pi) q[37];
U1q(3.058412894907952*pi,1.6731665811149479*pi) q[38];
U1q(0.713840726168223*pi,0.9841988411676708*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[18],q[1];
RZZ(0.5*pi) q[30],q[2];
RZZ(0.5*pi) q[3],q[29];
RZZ(0.5*pi) q[22],q[4];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[6],q[12];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[8],q[37];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[32],q[11];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[14],q[17];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[39],q[16];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[34],q[27];
RZZ(0.5*pi) q[36],q[38];
U1q(1.43279079959313*pi,1.2181090696772623*pi) q[0];
U1q(0.207396949712615*pi,0.28728480195698936*pi) q[1];
U1q(1.22449490319879*pi,0.6358134422613699*pi) q[2];
U1q(3.133006384195696*pi,0.7702657116355791*pi) q[3];
U1q(0.451727569745141*pi,1.368226376124376*pi) q[4];
U1q(1.95679259355947*pi,0.8252322715481556*pi) q[5];
U1q(1.192762765078*pi,1.9969960492165537*pi) q[6];
U1q(0.308160618397719*pi,1.4885313914956475*pi) q[7];
U1q(0.682810065543367*pi,1.301431518877937*pi) q[8];
U1q(1.03429350036698*pi,1.6322309938950976*pi) q[9];
U1q(0.171538184733897*pi,0.35843499938429346*pi) q[10];
U1q(1.82278265139275*pi,1.83836138804216*pi) q[11];
U1q(3.63431918618286*pi,1.9261035270294755*pi) q[12];
U1q(0.083867894075767*pi,1.442071576845418*pi) q[13];
U1q(0.381732318533313*pi,1.0704842139198547*pi) q[14];
U1q(0.576978394258021*pi,0.8497657761177564*pi) q[15];
U1q(0.461680811091589*pi,0.506999180633422*pi) q[16];
U1q(0.242783313509753*pi,1.8419462904616353*pi) q[17];
U1q(0.500187456101963*pi,1.1292335584898976*pi) q[18];
U1q(0.352686456726687*pi,0.1809025036382783*pi) q[19];
U1q(1.69666065616025*pi,1.8639767757795003*pi) q[20];
U1q(0.643196182648118*pi,1.0623665212171591*pi) q[21];
U1q(0.161574925866023*pi,1.5106706830726893*pi) q[22];
U1q(0.547419279331907*pi,0.4564078787453365*pi) q[23];
U1q(1.74736477991552*pi,1.3985383928901252*pi) q[24];
U1q(0.308750903209454*pi,0.9987673738415044*pi) q[25];
U1q(1.52749569437946*pi,0.2384663194823089*pi) q[26];
U1q(1.89553295359737*pi,0.29841017629824673*pi) q[27];
U1q(0.581224549609954*pi,0.6534279956774234*pi) q[28];
U1q(0.860093034458907*pi,1.9216685377930025*pi) q[29];
U1q(0.240175716960939*pi,0.12358459674138711*pi) q[30];
U1q(1.68310802591015*pi,0.5270445245837723*pi) q[31];
U1q(1.89403686860048*pi,1.637210144467879*pi) q[32];
U1q(1.19741966826888*pi,1.6373258273821412*pi) q[33];
U1q(0.456923995103911*pi,1.320221685853415*pi) q[34];
U1q(1.4176835608918*pi,0.6161354990251016*pi) q[35];
U1q(0.815320861566498*pi,1.7201162457698222*pi) q[36];
U1q(0.33147255242449*pi,1.0316560546790319*pi) q[37];
U1q(1.5622214272475*pi,1.9825996595961706*pi) q[38];
U1q(0.512684112039627*pi,1.3801854678190209*pi) q[39];
rz(2.7818909303227377*pi) q[0];
rz(1.7127151980430106*pi) q[1];
rz(3.36418655773863*pi) q[2];
rz(1.229734288364421*pi) q[3];
rz(2.631773623875624*pi) q[4];
rz(3.1747677284518443*pi) q[5];
rz(2.0030039507834463*pi) q[6];
rz(2.5114686085043525*pi) q[7];
rz(2.698568481122063*pi) q[8];
rz(0.3677690061049024*pi) q[9];
rz(1.6415650006157065*pi) q[10];
rz(2.16163861195784*pi) q[11];
rz(0.07389647297052448*pi) q[12];
rz(0.557928423154582*pi) q[13];
rz(2.9295157860801453*pi) q[14];
rz(1.1502342238822436*pi) q[15];
rz(3.493000819366578*pi) q[16];
rz(2.1580537095383647*pi) q[17];
rz(2.8707664415101024*pi) q[18];
rz(3.8190974963617217*pi) q[19];
rz(0.13602322422049973*pi) q[20];
rz(0.9376334787828409*pi) q[21];
rz(0.48932931692731074*pi) q[22];
rz(1.5435921212546635*pi) q[23];
rz(2.601461607109875*pi) q[24];
rz(3.0012326261584956*pi) q[25];
rz(1.761533680517691*pi) q[26];
rz(3.7015898237017533*pi) q[27];
rz(1.3465720043225766*pi) q[28];
rz(0.0783314622069975*pi) q[29];
rz(1.876415403258613*pi) q[30];
rz(1.4729554754162277*pi) q[31];
rz(2.362789855532121*pi) q[32];
rz(2.362674172617859*pi) q[33];
rz(0.679778314146585*pi) q[34];
rz(1.3838645009748984*pi) q[35];
rz(2.2798837542301778*pi) q[36];
rz(0.9683439453209681*pi) q[37];
rz(2.0174003404038294*pi) q[38];
rz(0.6198145321809791*pi) q[39];
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
