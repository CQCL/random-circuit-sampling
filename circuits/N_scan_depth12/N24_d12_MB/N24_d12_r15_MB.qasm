OPENQASM 2.0;
include "hqslib1.inc";

qreg q[24];
creg c[24];
U1q(1.88684344824472*pi,1.637274680609365*pi) q[0];
U1q(1.44972961940051*pi,0.07676741976525374*pi) q[1];
U1q(1.36443422218171*pi,0.5361875687340488*pi) q[2];
U1q(1.50392960492058*pi,1.4938099081954874*pi) q[3];
U1q(0.646459903903454*pi,1.692195445075461*pi) q[4];
U1q(0.510327459845538*pi,0.482334812195001*pi) q[5];
U1q(1.78721478221964*pi,1.4759017977136897*pi) q[6];
U1q(0.796543168781351*pi,0.0637888529781729*pi) q[7];
U1q(1.32573351035378*pi,0.49937098131071883*pi) q[8];
U1q(0.400933350212664*pi,1.285718666129932*pi) q[9];
U1q(1.67707741949616*pi,1.7042873572561956*pi) q[10];
U1q(3.683619539225081*pi,0.47767909305751366*pi) q[11];
U1q(1.32310059354913*pi,1.0347675843648045*pi) q[12];
U1q(0.211868249704936*pi,1.00473947603761*pi) q[13];
U1q(0.421321204873751*pi,1.76650696838076*pi) q[14];
U1q(1.48391505494984*pi,0.24275269562834573*pi) q[15];
U1q(0.344432426149639*pi,1.868744213774423*pi) q[16];
U1q(1.73246331388506*pi,0.9116960339994206*pi) q[17];
U1q(0.875536565874602*pi,1.39095330275474*pi) q[18];
U1q(1.31182921659105*pi,0.8991885526880798*pi) q[19];
U1q(0.383869343541418*pi,0.786099379238092*pi) q[20];
U1q(0.248569959870734*pi,0.357086109575887*pi) q[21];
U1q(1.33284812493722*pi,0.6309840316485423*pi) q[22];
U1q(1.4856585630052*pi,1.8480732923883105*pi) q[23];
RZZ(0.5*pi) q[0],q[10];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[2],q[9];
RZZ(0.5*pi) q[4],q[17];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[13],q[7];
RZZ(0.5*pi) q[19],q[8];
RZZ(0.5*pi) q[12],q[18];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[23],q[16];
RZZ(0.5*pi) q[21],q[22];
U1q(0.397061931908901*pi,0.4334493993539148*pi) q[0];
U1q(0.988054271557526*pi,1.683021291400474*pi) q[1];
U1q(0.470190464220523*pi,1.3722228524752889*pi) q[2];
U1q(0.431207768082735*pi,0.33375138093429735*pi) q[3];
U1q(0.67325333746237*pi,1.48599707983983*pi) q[4];
U1q(0.284026187058643*pi,0.5437487606281*pi) q[5];
U1q(0.214321810863431*pi,0.4979577713432326*pi) q[6];
U1q(0.715545157845542*pi,1.0843544462223602*pi) q[7];
U1q(0.602241095382751*pi,1.0094647343059688*pi) q[8];
U1q(0.293296880596964*pi,0.6606669300170598*pi) q[9];
U1q(0.74771027181887*pi,1.7788697030050056*pi) q[10];
U1q(0.475823483411268*pi,0.5167014090489837*pi) q[11];
U1q(0.382527208367516*pi,0.2671873912746845*pi) q[12];
U1q(0.138605829924049*pi,1.3595508685416702*pi) q[13];
U1q(0.296456949702179*pi,1.59879250856517*pi) q[14];
U1q(0.246917396612413*pi,1.4035538517976047*pi) q[15];
U1q(0.791659179131095*pi,1.3878615259353189*pi) q[16];
U1q(0.0556011924033482*pi,1.0461895964014407*pi) q[17];
U1q(0.170011738865594*pi,1.367208312380934*pi) q[18];
U1q(0.569012581099978*pi,0.2879068832259959*pi) q[19];
U1q(0.142959126687273*pi,0.7470062911270001*pi) q[20];
U1q(0.23871764257709*pi,1.70058022635232*pi) q[21];
U1q(0.250615917805944*pi,0.8350805127127323*pi) q[22];
U1q(0.327161022971268*pi,0.43277219661903055*pi) q[23];
RZZ(0.5*pi) q[0],q[7];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[22],q[2];
RZZ(0.5*pi) q[3],q[20];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[6],q[15];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[12],q[9];
RZZ(0.5*pi) q[23],q[10];
RZZ(0.5*pi) q[11],q[13];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[19],q[16];
U1q(0.395461987176752*pi,0.22262432896761464*pi) q[0];
U1q(0.562677686322425*pi,1.3554761843107936*pi) q[1];
U1q(0.92977107074478*pi,0.4502259455972286*pi) q[2];
U1q(0.898766468679167*pi,0.34723189848058755*pi) q[3];
U1q(0.672606545826458*pi,0.91641746141942*pi) q[4];
U1q(0.310308838453944*pi,1.9685186243677704*pi) q[5];
U1q(0.538736385592364*pi,1.1439197806759496*pi) q[6];
U1q(0.18561294951698*pi,0.9997746982106603*pi) q[7];
U1q(0.755136783391004*pi,0.004841485585818539*pi) q[8];
U1q(0.533895881257086*pi,1.4324970597827598*pi) q[9];
U1q(0.303307073746604*pi,1.5280728808347748*pi) q[10];
U1q(0.657062752154857*pi,0.5133323030982737*pi) q[11];
U1q(0.176342656029201*pi,0.14835499907413485*pi) q[12];
U1q(0.386188898962584*pi,1.5465637681822102*pi) q[13];
U1q(0.911197098719394*pi,0.10916864966586015*pi) q[14];
U1q(0.961927378215239*pi,1.248753256685636*pi) q[15];
U1q(0.663937941483045*pi,0.9706434107649602*pi) q[16];
U1q(0.842317994489703*pi,1.41428703725387*pi) q[17];
U1q(0.228611540206194*pi,0.23217373989684997*pi) q[18];
U1q(0.714283701202576*pi,0.9425758475725099*pi) q[19];
U1q(0.456592824164695*pi,0.11446064373765008*pi) q[20];
U1q(0.434974129000513*pi,0.051682919810160044*pi) q[21];
U1q(0.485830859031597*pi,1.185388484838012*pi) q[22];
U1q(0.90661428068855*pi,1.8936240423086304*pi) q[23];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[2],q[14];
RZZ(0.5*pi) q[22],q[3];
RZZ(0.5*pi) q[4],q[20];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[21],q[17];
RZZ(0.5*pi) q[19],q[18];
U1q(0.448946780154956*pi,0.3738087171212552*pi) q[0];
U1q(0.259846258868245*pi,1.7162166758174546*pi) q[1];
U1q(0.588472832001579*pi,1.7325401945205794*pi) q[2];
U1q(0.672592207096666*pi,1.9129457615702474*pi) q[3];
U1q(0.763274830684258*pi,0.4051312866336003*pi) q[4];
U1q(0.324761139243657*pi,0.5024258100534196*pi) q[5];
U1q(0.632987670532664*pi,1.8260705016840397*pi) q[6];
U1q(0.0646608818493718*pi,1.56754496817023*pi) q[7];
U1q(0.763712826640154*pi,0.013519769647277968*pi) q[8];
U1q(0.194182059292723*pi,0.21680734155867043*pi) q[9];
U1q(0.272188889666874*pi,0.15497664190992477*pi) q[10];
U1q(0.243579869522363*pi,1.2098466254245235*pi) q[11];
U1q(0.543175786560625*pi,1.4161142841889838*pi) q[12];
U1q(0.532475060733424*pi,1.2961455498462104*pi) q[13];
U1q(0.616770966768277*pi,0.2919791161245797*pi) q[14];
U1q(0.24562207548453*pi,0.8705351884786854*pi) q[15];
U1q(0.781798263316219*pi,1.1287326616431201*pi) q[16];
U1q(0.520427809533051*pi,0.19928646087037105*pi) q[17];
U1q(0.948110430449035*pi,1.24295923356663*pi) q[18];
U1q(0.470372759134416*pi,0.5960381175400595*pi) q[19];
U1q(0.548749727592004*pi,1.67190530768716*pi) q[20];
U1q(0.348911535104806*pi,0.44902887164152006*pi) q[21];
U1q(0.201669147227738*pi,1.194495571050262*pi) q[22];
U1q(0.580604821358026*pi,0.6862772814122104*pi) q[23];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[2],q[10];
RZZ(0.5*pi) q[4],q[13];
RZZ(0.5*pi) q[11],q[5];
RZZ(0.5*pi) q[6],q[9];
RZZ(0.5*pi) q[8],q[7];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[21],q[14];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[23],q[17];
U1q(0.685623366003153*pi,0.7640774271163657*pi) q[0];
U1q(0.423139746714216*pi,0.9432932491993533*pi) q[1];
U1q(0.798940166255905*pi,1.9925620870793193*pi) q[2];
U1q(0.163483889402795*pi,1.3713797571462667*pi) q[3];
U1q(0.54165481642101*pi,0.8839091517737696*pi) q[4];
U1q(0.577735948529912*pi,0.45501601970792027*pi) q[5];
U1q(0.342999513005407*pi,1.5997547005806396*pi) q[6];
U1q(0.550773663100816*pi,0.6968901875755407*pi) q[7];
U1q(0.978193461365649*pi,1.8270794221572189*pi) q[8];
U1q(0.566811593991097*pi,0.8996683539157004*pi) q[9];
U1q(0.210589627848175*pi,1.7560071027075548*pi) q[10];
U1q(0.546974052370742*pi,1.6758669450496742*pi) q[11];
U1q(0.895663005921647*pi,0.09886887224713448*pi) q[12];
U1q(0.685583246673746*pi,1.7836238709877996*pi) q[13];
U1q(0.576821092362586*pi,0.5486003292220403*pi) q[14];
U1q(0.672544674873026*pi,0.7327569349355656*pi) q[15];
U1q(0.632588139386348*pi,1.1814247516427798*pi) q[16];
U1q(0.337566010842629*pi,0.10198443914612199*pi) q[17];
U1q(0.658268843343102*pi,1.5199582973037309*pi) q[18];
U1q(0.418446593234851*pi,1.9616285049418396*pi) q[19];
U1q(0.153432446885097*pi,1.85220772469434*pi) q[20];
U1q(0.252074999506481*pi,0.25776618324143996*pi) q[21];
U1q(0.798222911225138*pi,0.28370642852734385*pi) q[22];
U1q(0.258082199803493*pi,0.19060999941634016*pi) q[23];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[6],q[5];
RZZ(0.5*pi) q[19],q[7];
RZZ(0.5*pi) q[8],q[18];
RZZ(0.5*pi) q[16],q[9];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[22],q[23];
U1q(0.89395193709543*pi,1.1266235186460651*pi) q[0];
U1q(0.365527634136367*pi,0.7710710926597546*pi) q[1];
U1q(0.528056174469564*pi,0.6916132719828187*pi) q[2];
U1q(0.476331877657922*pi,0.3973040768258578*pi) q[3];
U1q(0.363466787359925*pi,0.7172453435783996*pi) q[4];
U1q(0.677465551802407*pi,1.6244490785302492*pi) q[5];
U1q(0.423319467848479*pi,1.6593950839891303*pi) q[6];
U1q(0.670903624045135*pi,1.2855616132979009*pi) q[7];
U1q(0.678691371591608*pi,1.126365273821719*pi) q[8];
U1q(0.721377636760878*pi,1.4182009153023003*pi) q[9];
U1q(0.63420813206535*pi,0.896807593068095*pi) q[10];
U1q(0.337734106847369*pi,1.4259314026057233*pi) q[11];
U1q(0.584967344166276*pi,1.6207812790474048*pi) q[12];
U1q(0.30439421938147*pi,1.9734524318627003*pi) q[13];
U1q(0.843630280254113*pi,1.3663324961267005*pi) q[14];
U1q(0.758959460973591*pi,1.9795694587871449*pi) q[15];
U1q(0.552326388730625*pi,1.7018164458320992*pi) q[16];
U1q(0.467951098529909*pi,0.3727652270055213*pi) q[17];
U1q(0.0971558672995047*pi,1.4870874305493*pi) q[18];
U1q(0.302414827973707*pi,0.45517929570717897*pi) q[19];
U1q(0.428459146769269*pi,0.85587289126161*pi) q[20];
U1q(0.342670405164614*pi,1.0002539885844008*pi) q[21];
U1q(0.480689439049368*pi,1.5099935082481437*pi) q[22];
U1q(0.93176198874426*pi,0.9824651995287397*pi) q[23];
RZZ(0.5*pi) q[0],q[2];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[4],q[7];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[6],q[17];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[21],q[9];
RZZ(0.5*pi) q[18],q[10];
RZZ(0.5*pi) q[12],q[19];
RZZ(0.5*pi) q[14],q[13];
RZZ(0.5*pi) q[15],q[20];
U1q(0.409025749896814*pi,0.8878164626075655*pi) q[0];
U1q(0.707454219427338*pi,1.4882725666079537*pi) q[1];
U1q(0.666244951273153*pi,0.11768512712964885*pi) q[2];
U1q(0.434002193377414*pi,1.0166408095487878*pi) q[3];
U1q(0.320665576221625*pi,0.14763629732749983*pi) q[4];
U1q(0.874910730512553*pi,1.7532548087897997*pi) q[5];
U1q(0.814710855577021*pi,1.8259348856618196*pi) q[6];
U1q(0.337833213742385*pi,1.9407379547297996*pi) q[7];
U1q(0.626031903609844*pi,1.6926288258383195*pi) q[8];
U1q(0.643833492055357*pi,0.5796051293113997*pi) q[9];
U1q(0.68825249011143*pi,1.4517560024595948*pi) q[10];
U1q(0.811141192766229*pi,1.4415965589668147*pi) q[11];
U1q(0.576685175424296*pi,0.4423364313358036*pi) q[12];
U1q(0.248194030778323*pi,1.0451381297276008*pi) q[13];
U1q(0.521237090382697*pi,0.34160343043910046*pi) q[14];
U1q(0.573443516082544*pi,1.6513632730200456*pi) q[15];
U1q(0.611383262438124*pi,1.1064201044250996*pi) q[16];
U1q(0.822264878044543*pi,0.8459755701497205*pi) q[17];
U1q(0.292638636570867*pi,1.4913339169181992*pi) q[18];
U1q(0.546736713587935*pi,1.1649812819628806*pi) q[19];
U1q(0.362612095233764*pi,0.3056726238497003*pi) q[20];
U1q(0.140065509696118*pi,1.4924736447814997*pi) q[21];
U1q(0.65521959268127*pi,1.7740559618197427*pi) q[22];
U1q(0.553875417691701*pi,1.8556112271195797*pi) q[23];
rz(3.5416810173787336*pi) q[0];
rz(1.7837703115049468*pi) q[1];
rz(1.1544219979502515*pi) q[2];
rz(1.6602250457215124*pi) q[3];
rz(2.7259905102683994*pi) q[4];
rz(3.1488716914955006*pi) q[5];
rz(1.7851213990327413*pi) q[6];
rz(2.2877259734232993*pi) q[7];
rz(1.0512620160051824*pi) q[8];
rz(2.0801482751130003*pi) q[9];
rz(2.6826787678132042*pi) q[10];
rz(3.4600878366258865*pi) q[11];
rz(1.0792146608033963*pi) q[12];
rz(3.7858479480069*pi) q[13];
rz(2.928052956276*pi) q[14];
rz(3.2118240279111543*pi) q[15];
rz(0.7206384902941991*pi) q[16];
rz(1.9462964331922787*pi) q[17];
rz(2.9677131377143997*pi) q[18];
rz(0.08302864926131903*pi) q[19];
rz(0.6139891555793007*pi) q[20];
rz(2.6770414266862*pi) q[21];
rz(2.9779003950711562*pi) q[22];
rz(2.32514009383239*pi) q[23];
U1q(0.409025749896814*pi,1.429497479986256*pi) q[0];
U1q(1.70745421942734*pi,0.272042878112873*pi) q[1];
U1q(1.66624495127315*pi,0.272107125079859*pi) q[2];
U1q(0.434002193377414*pi,1.67686585527024*pi) q[3];
U1q(1.32066557622162*pi,1.873626807595943*pi) q[4];
U1q(1.87491073051255*pi,1.9021265002852816*pi) q[5];
U1q(1.81471085557702*pi,0.611056284694556*pi) q[6];
U1q(0.337833213742385*pi,1.228463928153106*pi) q[7];
U1q(0.626031903609844*pi,1.743890841843413*pi) q[8];
U1q(0.643833492055357*pi,1.65975340442442*pi) q[9];
U1q(1.68825249011143*pi,1.134434770272795*pi) q[10];
U1q(0.811141192766229*pi,1.9016843955926939*pi) q[11];
U1q(1.5766851754243*pi,0.521551092139281*pi) q[12];
U1q(3.248194030778323*pi,1.830986077734485*pi) q[13];
U1q(1.5212370903827*pi,0.269656386715139*pi) q[14];
U1q(1.57344351608254*pi,1.86318730093114*pi) q[15];
U1q(0.611383262438124*pi,0.8270585947193401*pi) q[16];
U1q(1.82226487804454*pi,1.79227200334193*pi) q[17];
U1q(3.292638636570867*pi,1.459047054632598*pi) q[18];
U1q(0.546736713587935*pi,0.248009931224169*pi) q[19];
U1q(0.362612095233764*pi,1.9196617794290185*pi) q[20];
U1q(1.14006550969612*pi,1.1695150714676599*pi) q[21];
U1q(0.65521959268127*pi,1.751956356890867*pi) q[22];
U1q(3.553875417691701*pi,1.180751320951922*pi) q[23];
RZZ(0.5*pi) q[0],q[2];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[4],q[7];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[6],q[17];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[21],q[9];
RZZ(0.5*pi) q[18],q[10];
RZZ(0.5*pi) q[12],q[19];
RZZ(0.5*pi) q[14],q[13];
RZZ(0.5*pi) q[15],q[20];
U1q(0.89395193709543*pi,1.66830453602474*pi) q[0];
U1q(3.634472365863632*pi,1.9892443520610763*pi) q[1];
U1q(3.471943825530436*pi,0.6981789802266531*pi) q[2];
U1q(0.476331877657922*pi,0.05752912254733*pi) q[3];
U1q(3.636533212640074*pi,0.30401776134504255*pi) q[4];
U1q(1.67746555180241*pi,1.030932230544799*pi) q[5];
U1q(1.42331946784848*pi,1.7775960863672498*pi) q[6];
U1q(1.67090362404514*pi,0.573287586721173*pi) q[7];
U1q(1.67869137159161*pi,0.17762728982682008*pi) q[8];
U1q(3.721377636760879*pi,0.4983491904153401*pi) q[9];
U1q(1.63420813206535*pi,0.6893831796642815*pi) q[10];
U1q(0.337734106847369*pi,0.88601923923161*pi) q[11];
U1q(3.415032655833724*pi,0.3431062444277021*pi) q[12];
U1q(3.30439421938147*pi,1.9026717755993863*pi) q[13];
U1q(3.156369719745887*pi,1.2449273210275362*pi) q[14];
U1q(1.75895946097359*pi,1.53498111516403*pi) q[15];
U1q(1.55232638873063*pi,1.42245493612636*pi) q[16];
U1q(1.46795109852991*pi,1.2654823464861065*pi) q[17];
U1q(3.097155867299504*pi,0.46329354100146536*pi) q[18];
U1q(1.30241482797371*pi,1.53820794496841*pi) q[19];
U1q(0.428459146769269*pi,0.46986204684089006*pi) q[20];
U1q(3.657329594835386*pi,0.6617347276646817*pi) q[21];
U1q(0.480689439049368*pi,0.48789390331932*pi) q[22];
U1q(3.06823801125574*pi,0.05389734854275963*pi) q[23];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[6],q[5];
RZZ(0.5*pi) q[19],q[7];
RZZ(0.5*pi) q[8],q[18];
RZZ(0.5*pi) q[16],q[9];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[22],q[23];
U1q(1.68562336600315*pi,1.3057584444951091*pi) q[0];
U1q(1.42313974671422*pi,0.8170221955214455*pi) q[1];
U1q(1.79894016625591*pi,0.397230165130144*pi) q[2];
U1q(1.16348388940279*pi,1.03160480286774*pi) q[3];
U1q(3.54165481642101*pi,0.13735395314968102*pi) q[4];
U1q(0.577735948529912*pi,0.8614991717224689*pi) q[5];
U1q(0.342999513005407*pi,0.7179557029587607*pi) q[6];
U1q(1.55077366310082*pi,0.16195901244349303*pi) q[7];
U1q(1.97819346136565*pi,1.4769131414912655*pi) q[8];
U1q(1.5668115939911*pi,0.016881751801948397*pi) q[9];
U1q(1.21058962784818*pi,1.5485826893037116*pi) q[10];
U1q(0.546974052370742*pi,1.135954781675556*pi) q[11];
U1q(1.89566300592165*pi,1.865018651227988*pi) q[12];
U1q(1.68558324667375*pi,0.7128432147244665*pi) q[13];
U1q(3.423178907637414*pi,0.06265948793221598*pi) q[14];
U1q(1.67254467487303*pi,1.2881685913124903*pi) q[15];
U1q(1.63258813938635*pi,0.9428466303157235*pi) q[16];
U1q(0.337566010842629*pi,1.9947015586267467*pi) q[17];
U1q(0.658268843343102*pi,0.49616440775588533*pi) q[18];
U1q(1.41844659323485*pi,1.0317587357337037*pi) q[19];
U1q(0.153432446885097*pi,0.46619688027362005*pi) q[20];
U1q(1.25207499950648*pi,0.40422253300768185*pi) q[21];
U1q(1.79822291122514*pi,1.26160682359846*pi) q[22];
U1q(1.25808219980349*pi,1.8457525486551636*pi) q[23];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[2],q[10];
RZZ(0.5*pi) q[4],q[13];
RZZ(0.5*pi) q[11],q[5];
RZZ(0.5*pi) q[6],q[9];
RZZ(0.5*pi) q[8],q[7];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[21],q[14];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[23],q[17];
U1q(3.551053219845043*pi,1.696027154490272*pi) q[0];
U1q(0.259846258868245*pi,0.5899456221395757*pi) q[1];
U1q(0.588472832001579*pi,0.1372082725714061*pi) q[2];
U1q(1.67259220709667*pi,1.4900387984437549*pi) q[3];
U1q(0.763274830684258*pi,1.658576088009501*pi) q[4];
U1q(0.324761139243657*pi,1.9089089620679691*pi) q[5];
U1q(1.63298767053266*pi,1.9442715040621597*pi) q[6];
U1q(0.0646608818493718*pi,0.032613793038179795*pi) q[7];
U1q(0.763712826640154*pi,0.6633534889813153*pi) q[8];
U1q(1.19418205929272*pi,0.3340207394449175*pi) q[9];
U1q(3.2721888896668743*pi,1.1496131501013451*pi) q[10];
U1q(1.24357986952236*pi,1.66993446205041*pi) q[11];
U1q(0.543175786560625*pi,1.182264063169838*pi) q[12];
U1q(1.53247506073342*pi,1.2003215358660295*pi) q[13];
U1q(3.383229033231723*pi,0.3192807010296761*pi) q[14];
U1q(3.75437792451547*pi,0.15039033776936228*pi) q[15];
U1q(1.78179826331622*pi,1.890154540316054*pi) q[16];
U1q(0.520427809533051*pi,1.0920035803509567*pi) q[17];
U1q(1.94811043044904*pi,1.2191653440187853*pi) q[18];
U1q(0.470372759134416*pi,1.666168348331924*pi) q[19];
U1q(0.548749727592004*pi,1.2858944632664402*pi) q[20];
U1q(1.34891153510481*pi,0.5954852214077619*pi) q[21];
U1q(3.201669147227738*pi,0.35081768107550904*pi) q[22];
U1q(1.58060482135803*pi,1.3414198306510237*pi) q[23];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[2],q[14];
RZZ(0.5*pi) q[22],q[3];
RZZ(0.5*pi) q[4],q[20];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[21],q[17];
RZZ(0.5*pi) q[19],q[18];
U1q(1.39546198717675*pi,0.8472115426439117*pi) q[0];
U1q(1.56267768632242*pi,0.2292051306329057*pi) q[1];
U1q(1.92977107074478*pi,0.8548940236480509*pi) q[2];
U1q(3.898766468679167*pi,0.924324935354095*pi) q[3];
U1q(0.672606545826458*pi,1.1698622627953208*pi) q[4];
U1q(0.310308838453944*pi,1.3750017763823208*pi) q[5];
U1q(1.53873638559236*pi,0.6264222250702498*pi) q[6];
U1q(0.18561294951698*pi,1.4648435230786099*pi) q[7];
U1q(1.755136783391*pi,0.6546752049198457*pi) q[8];
U1q(1.53389588125709*pi,1.11833102122083*pi) q[9];
U1q(1.3033070737466*pi,0.5227093890261951*pi) q[10];
U1q(1.65706275215486*pi,0.36644878437665884*pi) q[11];
U1q(1.1763426560292*pi,1.914504778054987*pi) q[12];
U1q(1.38618889896258*pi,1.4507397542020293*pi) q[13];
U1q(3.088802901280605*pi,0.5020911674883859*pi) q[14];
U1q(1.96192737821524*pi,0.7721722695624171*pi) q[15];
U1q(1.66393794148305*pi,0.04824379119420996*pi) q[16];
U1q(1.8423179944897*pi,0.3070041567344566*pi) q[17];
U1q(3.771388459793806*pi,1.2299508376885617*pi) q[18];
U1q(1.71428370120258*pi,0.012706078364364437*pi) q[19];
U1q(0.456592824164695*pi,0.7284497993169299*pi) q[20];
U1q(1.43497412900051*pi,0.9928311732391206*pi) q[21];
U1q(0.485830859031597*pi,1.341710594863259*pi) q[22];
U1q(1.90661428068855*pi,0.1340730697545971*pi) q[23];
RZZ(0.5*pi) q[0],q[7];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[22],q[2];
RZZ(0.5*pi) q[3],q[20];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[6],q[15];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[12],q[9];
RZZ(0.5*pi) q[23],q[10];
RZZ(0.5*pi) q[11],q[13];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[19],q[16];
U1q(1.3970619319089*pi,0.0580366130302119*pi) q[0];
U1q(3.011945728442482*pi,0.901660023543225*pi) q[1];
U1q(1.47019046422052*pi,1.932897116769999*pi) q[2];
U1q(3.431207768082735*pi,0.9378054529003839*pi) q[3];
U1q(1.67325333746237*pi,1.7394418812157308*pi) q[4];
U1q(0.284026187058643*pi,0.9502319126426491*pi) q[5];
U1q(3.214321810863431*pi,0.98046021573753*pi) q[6];
U1q(0.715545157845542*pi,0.5494232710903004*pi) q[7];
U1q(3.397758904617249*pi,1.6500519561996985*pi) q[8];
U1q(3.293296880596964*pi,0.3465008914551202*pi) q[9];
U1q(3.74771027181887*pi,0.27191256685597276*pi) q[10];
U1q(0.475823483411268*pi,0.3698178903273588*pi) q[11];
U1q(1.38252720836752*pi,1.7956723858544283*pi) q[12];
U1q(3.861394170075951*pi,0.6377526538425746*pi) q[13];
U1q(3.296456949702179*pi,1.0124673085890792*pi) q[14];
U1q(1.24691739661241*pi,1.9269728646743873*pi) q[15];
U1q(0.791659179131095*pi,0.4654619063645695*pi) q[16];
U1q(3.055601192403348*pi,1.6751015975868842*pi) q[17];
U1q(3.829988261134406*pi,1.094916265204482*pi) q[18];
U1q(3.4309874189000222*pi,1.6673750427108738*pi) q[19];
U1q(0.142959126687273*pi,1.36099544670628*pi) q[20];
U1q(0.23871764257709*pi,1.6417284797812801*pi) q[21];
U1q(0.250615917805944*pi,0.9914026227379686*pi) q[22];
U1q(0.327161022971268*pi,0.6732212240649869*pi) q[23];
RZZ(0.5*pi) q[0],q[10];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[2],q[9];
RZZ(0.5*pi) q[4],q[17];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[13],q[7];
RZZ(0.5*pi) q[19],q[8];
RZZ(0.5*pi) q[12],q[18];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[23],q[16];
RZZ(0.5*pi) q[21],q[22];
U1q(1.88684344824472*pi,0.8542113317747573*pi) q[0];
U1q(3.449729619400513*pi,1.5079138951784472*pi) q[1];
U1q(0.36443422218171*pi,1.096861833028764*pi) q[2];
U1q(0.503929604920583*pi,0.0978639801615735*pi) q[3];
U1q(1.64645990390345*pi,1.5332435159800983*pi) q[4];
U1q(0.510327459845538*pi,0.8888179642095482*pi) q[5];
U1q(1.78721478221964*pi,0.0025161893670744284*pi) q[6];
U1q(0.796543168781351*pi,1.52885767784612*pi) q[7];
U1q(1.32573351035378*pi,0.16014570919494986*pi) q[8];
U1q(1.40093335021266*pi,1.721449155342249*pi) q[9];
U1q(0.677077419496164*pi,0.1973302211071628*pi) q[10];
U1q(0.68361953922508*pi,0.330795574335899*pi) q[11];
U1q(0.323100593549126*pi,0.5632525789445384*pi) q[12];
U1q(1.21186824970494*pi,1.9925640463466276*pi) q[13];
U1q(0.421321204873751*pi,0.18018176840466227*pi) q[14];
U1q(1.48391505494984*pi,1.0877740208436446*pi) q[15];
U1q(0.344432426149639*pi,1.9463445942036701*pi) q[16];
U1q(0.73246331388506*pi,1.540608035184864*pi) q[17];
U1q(3.875536565874603*pi,0.07117127483067165*pi) q[18];
U1q(1.31182921659105*pi,1.0560933732487898*pi) q[19];
U1q(0.383869343541418*pi,0.40008853481737017*pi) q[20];
U1q(0.248569959870734*pi,1.2982343630048403*pi) q[21];
U1q(0.33284812493722*pi,1.7873061416737892*pi) q[22];
U1q(0.485658563005202*pi,1.0885223198342775*pi) q[23];
rz(3.1457886682252427*pi) q[0];
rz(2.492086104821553*pi) q[1];
rz(0.903138166971236*pi) q[2];
rz(3.9021360198384265*pi) q[3];
rz(0.46675648401990166*pi) q[4];
rz(1.1111820357904518*pi) q[5];
rz(3.9974838106329256*pi) q[6];
rz(0.47114232215387997*pi) q[7];
rz(3.83985429080505*pi) q[8];
rz(2.278550844657751*pi) q[9];
rz(3.802669778892837*pi) q[10];
rz(3.669204425664101*pi) q[11];
rz(1.4367474210554616*pi) q[12];
rz(2.0074359536533724*pi) q[13];
rz(1.8198182315953377*pi) q[14];
rz(2.9122259791563554*pi) q[15];
rz(2.05365540579633*pi) q[16];
rz(2.459391964815136*pi) q[17];
rz(3.9288287251693284*pi) q[18];
rz(2.94390662675121*pi) q[19];
rz(3.59991146518263*pi) q[20];
rz(2.7017656369951597*pi) q[21];
rz(2.212693858326211*pi) q[22];
rz(2.9114776801657225*pi) q[23];
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