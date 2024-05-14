OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.262741869543658*pi,1.340170274636756*pi) q[0];
U1q(0.815842837929437*pi,1.745069200493381*pi) q[1];
U1q(1.62363931499043*pi,0.9024248800324356*pi) q[2];
U1q(0.35385935577842*pi,0.91679518803731*pi) q[3];
U1q(0.574208637475705*pi,0.9775433216411*pi) q[4];
U1q(0.438257377034792*pi,0.66438473063432*pi) q[5];
U1q(1.4816103631485*pi,1.991179120843778*pi) q[6];
U1q(1.23561456080994*pi,1.9576448019632482*pi) q[7];
U1q(1.43271984451958*pi,0.8331328131060295*pi) q[8];
U1q(0.475830113577045*pi,1.1712664120235758*pi) q[9];
U1q(0.436306390076949*pi,1.184702385823331*pi) q[10];
U1q(3.850757404620048*pi,0.880179200597028*pi) q[11];
U1q(0.143811982934183*pi,0.96803644579516*pi) q[12];
U1q(3.793281168823174*pi,0.6944664515747926*pi) q[13];
U1q(0.425494282168502*pi,1.9851545453840187*pi) q[14];
U1q(0.507928089161505*pi,0.743655941577406*pi) q[15];
U1q(0.211790257901738*pi,1.130745518000075*pi) q[16];
U1q(1.12607365981299*pi,1.5170058358173963*pi) q[17];
U1q(0.914876000014347*pi,0.140204911960122*pi) q[18];
U1q(0.694928629619949*pi,1.879295281610801*pi) q[19];
U1q(1.4523749684979*pi,0.37040107253519544*pi) q[20];
U1q(0.180639339689423*pi,1.139721066666006*pi) q[21];
U1q(0.542889042637585*pi,0.413918797909617*pi) q[22];
U1q(0.183317730834113*pi,1.622573553445598*pi) q[23];
U1q(1.71568966385054*pi,1.8935394144470568*pi) q[24];
U1q(1.23367458754016*pi,1.7641685039238881*pi) q[25];
U1q(1.64855641154324*pi,1.9066049714151594*pi) q[26];
U1q(0.335733731524294*pi,1.828526765474044*pi) q[27];
U1q(0.577150319056268*pi,1.69638733419389*pi) q[28];
U1q(0.405890076692984*pi,1.805206064329268*pi) q[29];
U1q(1.16356271206957*pi,1.2389349670644685*pi) q[30];
U1q(1.3328846706789*pi,0.5637022985922969*pi) q[31];
U1q(0.0581590409277257*pi,1.388851092810403*pi) q[32];
U1q(1.40073754188637*pi,0.821783746547321*pi) q[33];
U1q(0.0437575856329392*pi,1.0678684930132412*pi) q[34];
U1q(0.114641279859539*pi,0.8948224144647801*pi) q[35];
U1q(0.432343880221001*pi,0.399386084407813*pi) q[36];
U1q(0.865369005137155*pi,0.931773927764923*pi) q[37];
U1q(0.600730944609502*pi,1.465918599621032*pi) q[38];
U1q(0.424013634565705*pi,0.280864796980755*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[16],q[1];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[5],q[8];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[28],q[12];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[14],q[39];
RZZ(0.5*pi) q[38],q[15];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[33],q[21];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[24],q[26];
RZZ(0.5*pi) q[37],q[27];
RZZ(0.5*pi) q[34],q[31];
U1q(0.99515750117092*pi,0.7922953530972401*pi) q[0];
U1q(0.365267925102105*pi,0.9310716897783*pi) q[1];
U1q(0.520439772681616*pi,0.7222959331222953*pi) q[2];
U1q(0.119849519537838*pi,1.36609045516947*pi) q[3];
U1q(0.0640409645640442*pi,0.21518233398710018*pi) q[4];
U1q(0.520471584856039*pi,1.9347827063209397*pi) q[5];
U1q(0.329494673958997*pi,1.238066823438508*pi) q[6];
U1q(0.844029616993208*pi,1.8775364944434978*pi) q[7];
U1q(0.341585495191213*pi,1.5791275906387297*pi) q[8];
U1q(0.356597758104902*pi,1.71121925446371*pi) q[9];
U1q(0.379673992630438*pi,1.7285646694764298*pi) q[10];
U1q(0.321253626539401*pi,1.5340977714619881*pi) q[11];
U1q(0.596753100081263*pi,0.2437419202794202*pi) q[12];
U1q(0.477170270148221*pi,0.21725951410430255*pi) q[13];
U1q(0.597940973478495*pi,0.54803876016959*pi) q[14];
U1q(0.788136330527384*pi,1.67982814079568*pi) q[15];
U1q(0.432745333963835*pi,0.70516850990193*pi) q[16];
U1q(0.462734905755178*pi,0.37648490772566645*pi) q[17];
U1q(0.873903532241235*pi,0.40838887723667994*pi) q[18];
U1q(0.468895434537916*pi,0.44691615964329*pi) q[19];
U1q(0.32314451161704*pi,1.7413700575739752*pi) q[20];
U1q(0.52324502699047*pi,0.33566088783831005*pi) q[21];
U1q(0.200352855772871*pi,0.13423380871022994*pi) q[22];
U1q(0.824145985694567*pi,0.028508897491509977*pi) q[23];
U1q(0.752258192757557*pi,1.0971337695018466*pi) q[24];
U1q(0.449144307152316*pi,0.7187018746674281*pi) q[25];
U1q(0.591171111718694*pi,0.975572959701819*pi) q[26];
U1q(0.113397069863332*pi,0.37786016017223*pi) q[27];
U1q(0.807244861747269*pi,0.5979150116884502*pi) q[28];
U1q(0.73408588460791*pi,1.4981140522241398*pi) q[29];
U1q(0.517813259762148*pi,1.3095574567618482*pi) q[30];
U1q(0.249473453205261*pi,0.298682566832007*pi) q[31];
U1q(0.547261853568499*pi,0.48081814881232*pi) q[32];
U1q(0.408257440417864*pi,1.86118881584397*pi) q[33];
U1q(0.236975699433746*pi,0.32013429962141005*pi) q[34];
U1q(0.517700765668463*pi,1.32237543184648*pi) q[35];
U1q(0.475739235069436*pi,0.9338402077852401*pi) q[36];
U1q(0.0713264112572142*pi,0.43349623641316004*pi) q[37];
U1q(0.461379569895977*pi,0.32144306252404986*pi) q[38];
U1q(0.690253582710354*pi,1.5351716930280879*pi) q[39];
RZZ(0.5*pi) q[0],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[27],q[3];
RZZ(0.5*pi) q[34],q[4];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[32],q[9];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[22],q[20];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[37],q[25];
RZZ(0.5*pi) q[33],q[38];
U1q(0.339350299746177*pi,0.23290412135382965*pi) q[0];
U1q(0.823028916185695*pi,1.4138781285379203*pi) q[1];
U1q(0.708167000603407*pi,1.059553318147925*pi) q[2];
U1q(0.497358847926872*pi,1.8457434398946404*pi) q[3];
U1q(0.551943689473758*pi,1.3048703384203497*pi) q[4];
U1q(0.603226392613796*pi,0.8711781535591099*pi) q[5];
U1q(0.341608299118192*pi,1.7402046946274377*pi) q[6];
U1q(0.700708588783108*pi,0.8599564565847881*pi) q[7];
U1q(0.463964929329521*pi,0.013745257026909208*pi) q[8];
U1q(0.28426397004489*pi,0.1993048025723798*pi) q[9];
U1q(0.463608326916557*pi,1.7614999729748604*pi) q[10];
U1q(0.480997685946156*pi,0.7687162821062579*pi) q[11];
U1q(0.378271561843199*pi,1.7179091728276603*pi) q[12];
U1q(0.376733401584643*pi,0.7700039932380323*pi) q[13];
U1q(0.559222034973992*pi,1.5789594466046397*pi) q[14];
U1q(0.59318312512159*pi,0.9158175904039503*pi) q[15];
U1q(0.589371527903631*pi,0.8504555023223901*pi) q[16];
U1q(0.428326685251116*pi,0.37098924814306633*pi) q[17];
U1q(0.44000713987878*pi,1.9424780303351303*pi) q[18];
U1q(0.580664958620604*pi,1.8265748441145497*pi) q[19];
U1q(0.56484049910886*pi,0.6093115685542658*pi) q[20];
U1q(0.677190279093963*pi,1.34617588042904*pi) q[21];
U1q(0.29822832687506*pi,1.5105451055185997*pi) q[22];
U1q(0.415869239155055*pi,1.3263116904502397*pi) q[23];
U1q(0.386198101459298*pi,1.9279065164289264*pi) q[24];
U1q(0.76832250493807*pi,0.16512142111324835*pi) q[25];
U1q(0.800592358957175*pi,1.3771933555446392*pi) q[26];
U1q(0.44999632119059*pi,1.5463313794270004*pi) q[27];
U1q(0.661612633226641*pi,0.40701024991156043*pi) q[28];
U1q(0.506771633226687*pi,0.7866751747162501*pi) q[29];
U1q(0.413816467576091*pi,0.8230945015115383*pi) q[30];
U1q(0.167353715806687*pi,0.9296417318930672*pi) q[31];
U1q(0.338749628430887*pi,0.03391605492875005*pi) q[32];
U1q(0.830080427849718*pi,0.01758293061372118*pi) q[33];
U1q(0.468153657898751*pi,1.563692288376*pi) q[34];
U1q(0.21059716972424*pi,1.1217181463869297*pi) q[35];
U1q(0.0623160836185685*pi,0.7402941163363597*pi) q[36];
U1q(0.578318682094189*pi,1.4740632349751501*pi) q[37];
U1q(0.698803405042294*pi,1.6889443032501097*pi) q[38];
U1q(0.397282283050981*pi,0.3630996192353899*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[5],q[30];
RZZ(0.5*pi) q[6],q[25];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[36],q[8];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[11],q[13];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[37],q[18];
RZZ(0.5*pi) q[22],q[19];
RZZ(0.5*pi) q[20],q[23];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[24],q[29];
RZZ(0.5*pi) q[28],q[38];
RZZ(0.5*pi) q[32],q[39];
RZZ(0.5*pi) q[34],q[35];
U1q(0.296002996858416*pi,1.37124456077916*pi) q[0];
U1q(0.614407832746199*pi,1.0840401167108098*pi) q[1];
U1q(0.24131495804153*pi,1.1448425632744152*pi) q[2];
U1q(0.3795523715904*pi,1.34305282313297*pi) q[3];
U1q(0.71815168296643*pi,0.5276208231027208*pi) q[4];
U1q(0.350481690771967*pi,1.8873897853990904*pi) q[5];
U1q(0.75424887391459*pi,1.8661584486108875*pi) q[6];
U1q(0.637368932244103*pi,0.4512437855778382*pi) q[7];
U1q(0.399786444011292*pi,0.004464103302559863*pi) q[8];
U1q(0.693717142531785*pi,1.8703060784318*pi) q[9];
U1q(0.277604571301298*pi,0.8331008373098898*pi) q[10];
U1q(0.674595172608123*pi,1.408040001907338*pi) q[11];
U1q(0.459513363998469*pi,0.007752636192280171*pi) q[12];
U1q(0.556330205345414*pi,1.4991293977729532*pi) q[13];
U1q(0.250964727228998*pi,1.00028269924797*pi) q[14];
U1q(0.383005148599471*pi,1.13329286515415*pi) q[15];
U1q(0.483653817287126*pi,1.5041876732517796*pi) q[16];
U1q(0.444841340794118*pi,1.5623679922906861*pi) q[17];
U1q(0.643927802343345*pi,0.87164835694215*pi) q[18];
U1q(0.1998111026652*pi,1.0784975214587007*pi) q[19];
U1q(0.207991866281117*pi,0.681744450763496*pi) q[20];
U1q(0.044811583734789*pi,0.6788889970286802*pi) q[21];
U1q(0.246346137490301*pi,0.7772758549973897*pi) q[22];
U1q(0.605902081899893*pi,1.5493931516055897*pi) q[23];
U1q(0.50271841387646*pi,0.7070556763013771*pi) q[24];
U1q(0.706776208924757*pi,0.4476189502331378*pi) q[25];
U1q(0.793258676957686*pi,0.5989410154075099*pi) q[26];
U1q(0.793542606374789*pi,0.7048911755215101*pi) q[27];
U1q(0.873952901357866*pi,0.27505601187418005*pi) q[28];
U1q(0.494437105862115*pi,1.9329088998095507*pi) q[29];
U1q(0.369738761350351*pi,0.31131473964023826*pi) q[30];
U1q(0.475320037223312*pi,1.2582893066905472*pi) q[31];
U1q(0.397872384434266*pi,0.37163494669987074*pi) q[32];
U1q(0.585403252918918*pi,1.9451217539680705*pi) q[33];
U1q(0.985144722532157*pi,0.7889537578468104*pi) q[34];
U1q(0.19510340080814*pi,1.6320652341767907*pi) q[35];
U1q(0.295935298972915*pi,0.053431537642450344*pi) q[36];
U1q(0.639909703496133*pi,1.2451651980057097*pi) q[37];
U1q(0.852071939509507*pi,0.48271188299426004*pi) q[38];
U1q(0.447063335263651*pi,1.7993084150022902*pi) q[39];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[30],q[1];
RZZ(0.5*pi) q[2],q[18];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[33],q[8];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[13],q[15];
RZZ(0.5*pi) q[24],q[14];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[17],q[25];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[27],q[31];
RZZ(0.5*pi) q[29],q[32];
RZZ(0.5*pi) q[38],q[36];
U1q(0.560312320316774*pi,0.9697704188491905*pi) q[0];
U1q(0.381778257045808*pi,1.9635865566218005*pi) q[1];
U1q(0.714148040169869*pi,0.8153594509459356*pi) q[2];
U1q(0.288829736994813*pi,0.060359189754480624*pi) q[3];
U1q(0.63257540256647*pi,1.0683885674070002*pi) q[4];
U1q(0.888634417990595*pi,1.4637958683111005*pi) q[5];
U1q(0.0742398116140532*pi,1.383684148051728*pi) q[6];
U1q(0.56416549011248*pi,0.12195986501914824*pi) q[7];
U1q(0.27157108829946*pi,1.6800720501722104*pi) q[8];
U1q(0.804870981217347*pi,1.3116348488447596*pi) q[9];
U1q(0.51379652630075*pi,1.94481257149058*pi) q[10];
U1q(0.730899771585432*pi,0.6858827763199287*pi) q[11];
U1q(0.655133054475643*pi,0.6664141511559993*pi) q[12];
U1q(0.486975513248153*pi,0.7368775861923922*pi) q[13];
U1q(0.615213841791169*pi,1.6342759335033996*pi) q[14];
U1q(0.416908961193711*pi,0.3831762244650996*pi) q[15];
U1q(0.237667248611493*pi,0.9266525942731008*pi) q[16];
U1q(0.393222758278711*pi,1.167686790347597*pi) q[17];
U1q(0.377134671777426*pi,1.26501806839243*pi) q[18];
U1q(0.635033600670667*pi,0.18556089184400015*pi) q[19];
U1q(0.349601020755502*pi,1.2570436722894947*pi) q[20];
U1q(0.576153264299417*pi,0.015876454791709627*pi) q[21];
U1q(0.0507168024324058*pi,1.3497742906710002*pi) q[22];
U1q(0.680239747719071*pi,1.6916165496668096*pi) q[23];
U1q(0.796200993028403*pi,1.417284946105747*pi) q[24];
U1q(0.188694983873085*pi,0.7638679875280996*pi) q[25];
U1q(0.26925995395599*pi,1.7650244835913593*pi) q[26];
U1q(0.487393894227957*pi,0.9167710252103003*pi) q[27];
U1q(0.538502487720789*pi,1.9865783863191009*pi) q[28];
U1q(0.127068076560605*pi,0.1400361876264995*pi) q[29];
U1q(0.677194611196329*pi,1.7678917118133484*pi) q[30];
U1q(0.531812537474692*pi,0.8756912262144869*pi) q[31];
U1q(0.793303464288472*pi,1.3840047928184998*pi) q[32];
U1q(0.339952774520706*pi,1.4384734542203415*pi) q[33];
U1q(0.648828620964607*pi,0.19573815372499936*pi) q[34];
U1q(0.645966251838841*pi,1.2601059249568003*pi) q[35];
U1q(0.68316139206257*pi,0.34842482470102*pi) q[36];
U1q(0.633998485008204*pi,1.6708231116794696*pi) q[37];
U1q(0.563790314680445*pi,1.2427855421375398*pi) q[38];
U1q(0.758772674356295*pi,1.84597729444015*pi) q[39];
RZZ(0.5*pi) q[10],q[0];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[11],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[26],q[16];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[27],q[18];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[21],q[20];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[34],q[33];
RZZ(0.5*pi) q[39],q[36];
U1q(0.638021028504934*pi,1.8093527057877008*pi) q[0];
U1q(0.622000384694257*pi,1.7105128620415009*pi) q[1];
U1q(0.669048723484535*pi,0.46577169201173696*pi) q[2];
U1q(0.50018818627496*pi,1.2015190866866998*pi) q[3];
U1q(0.727816478614912*pi,0.8858043165843004*pi) q[4];
U1q(0.564761588871278*pi,1.1778999675565007*pi) q[5];
U1q(0.47919648855137*pi,0.9878018449813784*pi) q[6];
U1q(0.465719395407811*pi,0.4216888075806473*pi) q[7];
U1q(0.525552381679995*pi,1.56394616015813*pi) q[8];
U1q(0.543294444724757*pi,1.5075817538427998*pi) q[9];
U1q(0.821597232838093*pi,0.4551171631298008*pi) q[10];
U1q(0.668053036654673*pi,0.8724830356940281*pi) q[11];
U1q(0.379654453400269*pi,1.3314166555170992*pi) q[12];
U1q(0.598801612778942*pi,1.7525504691429923*pi) q[13];
U1q(0.579291253960021*pi,1.9465229963593007*pi) q[14];
U1q(0.483297972496696*pi,0.5231708642458006*pi) q[15];
U1q(0.466130840723622*pi,1.7308125985959002*pi) q[16];
U1q(0.463295867327897*pi,1.424715188043697*pi) q[17];
U1q(0.202355668510097*pi,0.9500821814795692*pi) q[18];
U1q(0.721346786747203*pi,1.6612788769276001*pi) q[19];
U1q(0.474440113505498*pi,0.46416319177799537*pi) q[20];
U1q(0.237062052168202*pi,0.73209350678008*pi) q[21];
U1q(0.593232607416372*pi,1.9629389951237997*pi) q[22];
U1q(0.414437148383012*pi,0.041690604641999585*pi) q[23];
U1q(0.293408364277754*pi,0.49049523350065627*pi) q[24];
U1q(0.256160373353737*pi,1.7015568874883886*pi) q[25];
U1q(0.344662693110749*pi,1.3848290140690604*pi) q[26];
U1q(0.738557496690321*pi,1.1573288479096*pi) q[27];
U1q(0.339055549741701*pi,0.007548100758800302*pi) q[28];
U1q(0.560199266932443*pi,0.020200935974100886*pi) q[29];
U1q(0.473899695152778*pi,1.1499089780596687*pi) q[30];
U1q(0.560211416562315*pi,0.3615635367779966*pi) q[31];
U1q(0.144086142657157*pi,1.9510296558572993*pi) q[32];
U1q(0.826447857870534*pi,0.8399450911600308*pi) q[33];
U1q(0.58990899265821*pi,1.1828842808514004*pi) q[34];
U1q(0.638869034632029*pi,1.1283060460082002*pi) q[35];
U1q(0.735891728581672*pi,0.45491221341660015*pi) q[36];
U1q(0.844607726541788*pi,0.6748184991041999*pi) q[37];
U1q(0.842200881856885*pi,0.3781319426033498*pi) q[38];
U1q(0.828355997450689*pi,0.04823887464749976*pi) q[39];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[2],q[25];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[4],q[31];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[39],q[9];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[12],q[15];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[34],q[16];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[21],q[38];
RZZ(0.5*pi) q[35],q[22];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[26],q[29];
U1q(0.374472180800264*pi,0.26758256851329953*pi) q[0];
U1q(0.330158671718973*pi,0.6272647692467004*pi) q[1];
U1q(0.461769544035561*pi,1.2035572342161363*pi) q[2];
U1q(0.534610432193195*pi,1.6935091268838*pi) q[3];
U1q(0.928109578732113*pi,1.5815633512315*pi) q[4];
U1q(0.5474421234045*pi,1.0899860503382008*pi) q[5];
U1q(0.472972750228336*pi,1.0059844908836784*pi) q[6];
U1q(0.746503563812168*pi,0.05469266681014773*pi) q[7];
U1q(0.769045079854107*pi,0.1987580607567292*pi) q[8];
U1q(0.265285481923285*pi,0.6676423560943991*pi) q[9];
U1q(0.442679454815926*pi,1.2196186799111999*pi) q[10];
U1q(0.656219866334921*pi,0.9393470091895288*pi) q[11];
U1q(0.214625877696613*pi,0.9684735092165013*pi) q[12];
U1q(0.475918088732223*pi,0.9738369138157914*pi) q[13];
U1q(0.102603203968405*pi,0.977565095573901*pi) q[14];
U1q(0.621633844100443*pi,1.1345087317333*pi) q[15];
U1q(0.494677370276327*pi,0.5097503876428995*pi) q[16];
U1q(0.229655237530126*pi,1.666818490774796*pi) q[17];
U1q(0.686348345477643*pi,1.7942227415095005*pi) q[18];
U1q(0.524466283507217*pi,0.1087273082614999*pi) q[19];
U1q(0.784791493022655*pi,1.4826127598571954*pi) q[20];
U1q(0.463677149801024*pi,0.9117754588547999*pi) q[21];
U1q(0.574583811275911*pi,0.05800770073659933*pi) q[22];
U1q(0.874603212430059*pi,0.9286541382035995*pi) q[23];
U1q(0.377962012689136*pi,1.295721194997956*pi) q[24];
U1q(0.598123313499557*pi,1.5906141218959888*pi) q[25];
U1q(0.375999240637975*pi,0.33602825339135833*pi) q[26];
U1q(0.119944963785751*pi,1.4160995977606987*pi) q[27];
U1q(0.725605971797857*pi,0.4542388978199998*pi) q[28];
U1q(0.453704755888201*pi,0.08028010379149997*pi) q[29];
U1q(0.912945365000089*pi,1.1556537065035677*pi) q[30];
U1q(0.617896422428692*pi,0.5408745296061976*pi) q[31];
U1q(0.308816131011022*pi,0.19763399003289983*pi) q[32];
U1q(0.65827744091697*pi,0.7523757877785222*pi) q[33];
U1q(0.706283297344539*pi,0.8051371352010008*pi) q[34];
U1q(0.344263226437629*pi,0.5668354551593993*pi) q[35];
U1q(0.232270528733978*pi,1.3321554580740997*pi) q[36];
U1q(0.367245256034255*pi,1.4463938237503005*pi) q[37];
U1q(0.293619389695497*pi,1.0122977846667993*pi) q[38];
U1q(0.422470240569498*pi,0.9901863234687003*pi) q[39];
rz(3.4390504103128006*pi) q[0];
rz(2.8086430954811004*pi) q[1];
rz(3.3474883089111636*pi) q[2];
rz(1.0265739665624984*pi) q[3];
rz(3.7763548158596993*pi) q[4];
rz(2.8097836277690007*pi) q[5];
rz(2.4381545467374224*pi) q[6];
rz(3.946475068471951*pi) q[7];
rz(3.6266870765658705*pi) q[8];
rz(3.0995379003009997*pi) q[9];
rz(1.6392973731817992*pi) q[10];
rz(2.6752502713330717*pi) q[11];
rz(0.5280115112006989*pi) q[12];
rz(3.6318407283244074*pi) q[13];
rz(0.7182725409809017*pi) q[14];
rz(2.3452198871360004*pi) q[15];
rz(3.1666858807611007*pi) q[16];
rz(2.0171177329399033*pi) q[17];
rz(1.1935555050822*pi) q[18];
rz(0.27818599936840016*pi) q[19];
rz(3.1073875565472058*pi) q[20];
rz(2.4915608106686005*pi) q[21];
rz(0.4514729337685992*pi) q[22];
rz(1.3269075062046003*pi) q[23];
rz(1.8468922894808433*pi) q[24];
rz(3.534607957864811*pi) q[25];
rz(1.1224777433628397*pi) q[26];
rz(3.5454027973818008*pi) q[27];
rz(3.4855417263306006*pi) q[28];
rz(3.4570567167759005*pi) q[29];
rz(3.176437179309932*pi) q[30];
rz(1.126520747114803*pi) q[31];
rz(0.07383889150959888*pi) q[32];
rz(3.1118549054036784*pi) q[33];
rz(2.3358750497834*pi) q[34];
rz(0.38106428134339865*pi) q[35];
rz(2.0331556129731005*pi) q[36];
rz(2.096987051778701*pi) q[37];
rz(3.5725269905027*pi) q[38];
rz(2.1682407027847006*pi) q[39];
U1q(0.374472180800264*pi,0.706632978826052*pi) q[0];
U1q(0.330158671718973*pi,0.4359078647278001*pi) q[1];
U1q(0.461769544035561*pi,1.551045543127324*pi) q[2];
U1q(1.5346104321932*pi,1.72008309344631*pi) q[3];
U1q(0.928109578732113*pi,0.357918167091122*pi) q[4];
U1q(0.5474421234045*pi,0.899769678107209*pi) q[5];
U1q(1.47297275022834*pi,0.44413903762111*pi) q[6];
U1q(1.74650356381217*pi,1.00116773528207*pi) q[7];
U1q(1.76904507985411*pi,0.825445137322558*pi) q[8];
U1q(3.265285481923285*pi,0.7671802563953201*pi) q[9];
U1q(1.44267945481593*pi,1.85891605309293*pi) q[10];
U1q(0.656219866334921*pi,0.614597280522577*pi) q[11];
U1q(0.214625877696613*pi,0.49648502041715004*pi) q[12];
U1q(0.475918088732223*pi,1.60567764214021*pi) q[13];
U1q(0.102603203968405*pi,0.695837636554804*pi) q[14];
U1q(0.621633844100443*pi,0.479728618869274*pi) q[15];
U1q(0.494677370276327*pi,0.67643626840392*pi) q[16];
U1q(1.22965523753013*pi,0.683936223714649*pi) q[17];
U1q(0.686348345477643*pi,1.9877782465916343*pi) q[18];
U1q(3.524466283507217*pi,1.386913307629936*pi) q[19];
U1q(1.78479149302266*pi,1.5900003164043919*pi) q[20];
U1q(1.46367714980102*pi,0.40333626952338*pi) q[21];
U1q(1.57458381127591*pi,1.509480634505235*pi) q[22];
U1q(1.87460321243006*pi,1.25556164440821*pi) q[23];
U1q(3.377962012689136*pi,0.142613484478857*pi) q[24];
U1q(1.59812331349956*pi,0.125222079760869*pi) q[25];
U1q(0.375999240637975*pi,0.458505996754206*pi) q[26];
U1q(1.11994496378575*pi,1.9615023951424928*pi) q[27];
U1q(1.72560597179786*pi,0.939780624150623*pi) q[28];
U1q(0.453704755888201*pi,0.537336820567409*pi) q[29];
U1q(1.91294536500009*pi,1.332090885813503*pi) q[30];
U1q(1.61789642242869*pi,0.667395276720979*pi) q[31];
U1q(0.308816131011022*pi,1.271472881542552*pi) q[32];
U1q(0.65827744091697*pi,0.8642306931822199*pi) q[33];
U1q(1.70628329734454*pi,0.141012184984382*pi) q[34];
U1q(0.344263226437629*pi,1.9478997365027686*pi) q[35];
U1q(1.23227052873398*pi,0.365311071047274*pi) q[36];
U1q(1.36724525603425*pi,0.5433808755290499*pi) q[37];
U1q(1.2936193896955*pi,1.584824775169465*pi) q[38];
U1q(0.422470240569498*pi,0.158427026253396*pi) q[39];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[2],q[25];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[4],q[31];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[39],q[9];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[12],q[15];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[34],q[16];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[21],q[38];
RZZ(0.5*pi) q[35],q[22];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[26],q[29];
U1q(0.638021028504934*pi,0.248403116100479*pi) q[0];
U1q(0.622000384694257*pi,1.5191559575226399*pi) q[1];
U1q(0.669048723484535*pi,0.81326000092296*pi) q[2];
U1q(3.499811813725039*pi,1.2120731336433708*pi) q[3];
U1q(0.727816478614912*pi,1.662159132443944*pi) q[4];
U1q(0.564761588871278*pi,0.987683595325482*pi) q[5];
U1q(3.479196488551369*pi,1.4623216835234696*pi) q[6];
U1q(1.46571939540781*pi,1.6341715945115083*pi) q[7];
U1q(1.52555238168*pi,1.4602570379211295*pi) q[8];
U1q(1.54329444472476*pi,1.9272408586468794*pi) q[9];
U1q(3.178402767161907*pi,0.6234175698742757*pi) q[10];
U1q(1.66805303665467*pi,1.54773330702713*pi) q[11];
U1q(0.379654453400269*pi,1.8594281667177999*pi) q[12];
U1q(3.598801612778942*pi,0.384391197467346*pi) q[13];
U1q(1.57929125396002*pi,0.6647955373402601*pi) q[14];
U1q(0.483297972496696*pi,0.86839075138174*pi) q[15];
U1q(1.46613084072362*pi,1.897498479357*pi) q[16];
U1q(3.536704132672103*pi,0.9260395264457193*pi) q[17];
U1q(0.202355668510097*pi,1.1436376865617501*pi) q[18];
U1q(1.7213467867472*pi,0.8343617389638376*pi) q[19];
U1q(3.525559886494503*pi,0.6084498844836179*pi) q[20];
U1q(1.2370620521682*pi,1.5830182215980508*pi) q[21];
U1q(3.406767392583629*pi,0.6045493401180555*pi) q[22];
U1q(1.41443714838301*pi,0.14252517796987796*pi) q[23];
U1q(3.293408364277754*pi,0.9478394459761859*pi) q[24];
U1q(3.7438396266462632*pi,1.0142793141684585*pi) q[25];
U1q(3.344662693110749*pi,0.5073067574318899*pi) q[26];
U1q(1.73855749669032*pi,0.2202731449935203*pi) q[27];
U1q(3.660944450258299*pi,1.3864714212118687*pi) q[28];
U1q(0.560199266932443*pi,1.4772576527499899*pi) q[29];
U1q(3.526100304847222*pi,1.3378356142573504*pi) q[30];
U1q(3.439788583437685*pi,1.8467062695491683*pi) q[31];
U1q(1.14408614265716*pi,1.0248685473669101*pi) q[32];
U1q(0.826447857870534*pi,0.9517999965636901*pi) q[33];
U1q(1.58990899265821*pi,1.7632650393340108*pi) q[34];
U1q(1.63886903463203*pi,0.50937032735151*pi) q[35];
U1q(3.264108271418328*pi,0.24255431570483316*pi) q[36];
U1q(1.84460772654179*pi,1.314956200175156*pi) q[37];
U1q(1.84220088185689*pi,1.218990617232901*pi) q[38];
U1q(0.828355997450689*pi,1.2164795774321702*pi) q[39];
RZZ(0.5*pi) q[10],q[0];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[11],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[26],q[16];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[27],q[18];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[21],q[20];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[34],q[33];
RZZ(0.5*pi) q[39],q[36];
U1q(0.560312320316774*pi,1.4088208291619901*pi) q[0];
U1q(0.381778257045808*pi,0.7722296521029199*pi) q[1];
U1q(0.714148040169869*pi,1.162847759857135*pi) q[2];
U1q(3.288829736994813*pi,1.3532330305756122*pi) q[3];
U1q(1.63257540256647*pi,1.84474338326662*pi) q[4];
U1q(0.888634417990595*pi,0.273579496080041*pi) q[5];
U1q(1.07423981161405*pi,1.8582039865938493*pi) q[6];
U1q(0.56416549011248*pi,1.334442651949986*pi) q[7];
U1q(1.27157108829946*pi,1.5763829279351866*pi) q[8];
U1q(0.804870981217347*pi,0.7312939536488492*pi) q[9];
U1q(3.48620347369925*pi,1.1337221615135054*pi) q[10];
U1q(1.73089977158543*pi,1.734333566401272*pi) q[11];
U1q(0.655133054475643*pi,0.1944256623566698*pi) q[12];
U1q(3.513024486751847*pi,1.4000640804179147*pi) q[13];
U1q(1.61521384179117*pi,0.977042600196186*pi) q[14];
U1q(0.416908961193711*pi,0.72839611160106*pi) q[15];
U1q(3.762332751388507*pi,0.7016584836798412*pi) q[16];
U1q(3.393222758278711*pi,1.183067924141847*pi) q[17];
U1q(1.37713467177743*pi,0.4585735734746099*pi) q[18];
U1q(0.635033600670667*pi,0.35864375388026737*pi) q[19];
U1q(3.349601020755502*pi,0.8155694039720982*pi) q[20];
U1q(0.576153264299417*pi,0.8668011696096778*pi) q[21];
U1q(3.9492831975675964*pi,0.21771404457085364*pi) q[22];
U1q(1.68023974771907*pi,0.7924511229947377*pi) q[23];
U1q(1.7962009930284*pi,0.874629158581284*pi) q[24];
U1q(1.18869498387309*pi,0.9519682141287766*pi) q[25];
U1q(1.26925995395599*pi,1.127111287909607*pi) q[26];
U1q(0.487393894227957*pi,0.9797153222941375*pi) q[27];
U1q(3.538502487720789*pi,1.4074411356515828*pi) q[28];
U1q(0.127068076560605*pi,0.5970929044023898*pi) q[29];
U1q(1.67719461119633*pi,0.7198528805036994*pi) q[30];
U1q(1.53181253747469*pi,1.3325785801126573*pi) q[31];
U1q(1.79330346428847*pi,1.5918934104056763*pi) q[32];
U1q(0.339952774520706*pi,0.5503283596240003*pi) q[33];
U1q(0.648828620964607*pi,1.7761189122076289*pi) q[34];
U1q(3.354033748161159*pi,1.3775704484028473*pi) q[35];
U1q(3.31683860793743*pi,0.3490417044203811*pi) q[36];
U1q(0.633998485008204*pi,1.3109608127504258*pi) q[37];
U1q(1.56379031468045*pi,1.083644216767081*pi) q[38];
U1q(0.758772674356295*pi,1.01421799722482*pi) q[39];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[30],q[1];
RZZ(0.5*pi) q[2],q[18];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[33],q[8];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[13],q[15];
RZZ(0.5*pi) q[24],q[14];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[17],q[25];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[27],q[31];
RZZ(0.5*pi) q[29],q[32];
RZZ(0.5*pi) q[38],q[36];
U1q(0.296002996858416*pi,1.8102949710919498*pi) q[0];
U1q(0.614407832746199*pi,0.8926832121919404*pi) q[1];
U1q(0.24131495804153*pi,1.4923308721856001*pi) q[2];
U1q(0.3795523715904*pi,0.6359266639540921*pi) q[3];
U1q(3.28184831703357*pi,1.3855111275708678*pi) q[4];
U1q(0.350481690771967*pi,1.6971734131680698*pi) q[5];
U1q(3.24575112608541*pi,1.3757296860346955*pi) q[6];
U1q(1.6373689322441*pi,0.6637265725086792*pi) q[7];
U1q(3.600213555988707*pi,0.25199087480483584*pi) q[8];
U1q(1.69371714253178*pi,0.2899651832358794*pi) q[9];
U1q(1.2776045713013*pi,1.2454338956941884*pi) q[10];
U1q(0.674595172608123*pi,1.456490791988709*pi) q[11];
U1q(3.459513363998469*pi,1.5357641473929604*pi) q[12];
U1q(3.443669794654586*pi,0.6378122688373549*pi) q[13];
U1q(3.250964727228998*pi,0.34304936594074587*pi) q[14];
U1q(1.38300514859947*pi,1.478512752290108*pi) q[15];
U1q(1.48365381728713*pi,1.1241234047011615*pi) q[16];
U1q(1.44484134079412*pi,0.5777491260849352*pi) q[17];
U1q(1.64392780234335*pi,0.851943284924888*pi) q[18];
U1q(0.1998111026652*pi,0.25158038349491774*pi) q[19];
U1q(0.207991866281117*pi,1.2402701824460785*pi) q[20];
U1q(3.044811583734789*pi,1.5298137118466437*pi) q[21];
U1q(3.246346137490301*pi,0.7902124802444606*pi) q[22];
U1q(3.394097918100107*pi,1.934674521055961*pi) q[23];
U1q(3.50271841387646*pi,1.5848584283856488*pi) q[24];
U1q(0.706776208924757*pi,1.6357191768338137*pi) q[25];
U1q(1.79325867695769*pi,0.9610278197257873*pi) q[26];
U1q(1.79354260637479*pi,1.7678354726053978*pi) q[27];
U1q(0.873952901357866*pi,0.6959187612066833*pi) q[28];
U1q(0.494437105862115*pi,1.3899656165854397*pi) q[29];
U1q(1.36973876135035*pi,0.26327590833058334*pi) q[30];
U1q(1.47532003722331*pi,0.7151766605887204*pi) q[31];
U1q(0.397872384434266*pi,1.5795235642870455*pi) q[32];
U1q(1.58540325291892*pi,1.0569766593717302*pi) q[33];
U1q(0.985144722532157*pi,0.36933451632943903*pi) q[34];
U1q(3.80489659919186*pi,1.0056111391828753*pi) q[35];
U1q(3.704064701027085*pi,1.644034991478952*pi) q[36];
U1q(0.639909703496133*pi,1.8853028990766658*pi) q[37];
U1q(3.147928060490494*pi,1.8437178759103547*pi) q[38];
U1q(1.44706333526365*pi,1.9675491177869597*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[5],q[30];
RZZ(0.5*pi) q[6],q[25];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[36],q[8];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[11],q[13];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[37],q[18];
RZZ(0.5*pi) q[22],q[19];
RZZ(0.5*pi) q[20],q[23];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[24],q[29];
RZZ(0.5*pi) q[28],q[38];
RZZ(0.5*pi) q[32],q[39];
RZZ(0.5*pi) q[34],q[35];
U1q(0.339350299746177*pi,0.6719545316666302*pi) q[0];
U1q(3.823028916185695*pi,1.2225212240190402*pi) q[1];
U1q(0.708167000603407*pi,1.40704162705911*pi) q[2];
U1q(0.497358847926872*pi,1.1386172807157617*pi) q[3];
U1q(3.448056310526242*pi,0.6082616122532376*pi) q[4];
U1q(0.603226392613796*pi,0.6809617813280899*pi) q[5];
U1q(3.341608299118192*pi,0.5016834400181445*pi) q[6];
U1q(3.299291411216892*pi,0.25501390150172654*pi) q[7];
U1q(3.463964929329521*pi,0.2427097210804785*pi) q[8];
U1q(1.28426397004489*pi,1.9609664590952933*pi) q[9];
U1q(1.46360832691656*pi,0.17383303135915829*pi) q[10];
U1q(0.480997685946156*pi,0.8171670721876318*pi) q[11];
U1q(3.378271561843199*pi,0.8256076107575732*pi) q[12];
U1q(1.37673340158464*pi,0.36693767337227756*pi) q[13];
U1q(3.440777965026008*pi,0.7643726185840678*pi) q[14];
U1q(1.59318312512159*pi,0.6959880270403072*pi) q[15];
U1q(1.58937152790363*pi,0.47039123377177194*pi) q[16];
U1q(1.42832668525112*pi,1.769127870232559*pi) q[17];
U1q(0.44000713987878*pi,0.9227729583178581*pi) q[18];
U1q(0.580664958620604*pi,1.999657706150768*pi) q[19];
U1q(0.56484049910886*pi,1.1678373002368483*pi) q[20];
U1q(3.322809720906037*pi,0.8625268284462853*pi) q[21];
U1q(0.29822832687506*pi,0.5234817307656705*pi) q[22];
U1q(3.584130760844944*pi,0.15775598221131104*pi) q[23];
U1q(1.3861981014593*pi,1.8057092685131941*pi) q[24];
U1q(0.76832250493807*pi,1.3532216477139256*pi) q[25];
U1q(3.199407641042825*pi,0.1827754795886518*pi) q[26];
U1q(1.44999632119059*pi,0.9263952686999186*pi) q[27];
U1q(3.6616126332266408*pi,0.8278729992440628*pi) q[28];
U1q(0.506771633226687*pi,1.2437318914921391*pi) q[29];
U1q(1.41381646757609*pi,1.7514961464592875*pi) q[30];
U1q(1.16735371580669*pi,0.04382423538620395*pi) q[31];
U1q(1.33874962843089*pi,1.2418046725159364*pi) q[32];
U1q(3.169919572150283*pi,1.9845154827260778*pi) q[33];
U1q(0.468153657898751*pi,1.1440730468586287*pi) q[34];
U1q(3.789402830275761*pi,1.515958226972732*pi) q[35];
U1q(3.9376839163814297*pi,1.957172412785042*pi) q[36];
U1q(0.578318682094189*pi,1.1142009360461058*pi) q[37];
U1q(3.3011965949577062*pi,0.6374854556545047*pi) q[38];
U1q(1.39728228305098*pi,1.4037579135538643*pi) q[39];
RZZ(0.5*pi) q[0],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[27],q[3];
RZZ(0.5*pi) q[34],q[4];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[32],q[9];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[22],q[20];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[37],q[25];
RZZ(0.5*pi) q[33],q[38];
U1q(0.99515750117092*pi,1.2313457634100402*pi) q[0];
U1q(1.36526792510211*pi,1.7053276627786529*pi) q[1];
U1q(1.52043977268162*pi,0.06978424203347*pi) q[2];
U1q(0.119849519537838*pi,1.658964295990602*pi) q[3];
U1q(3.9359590354359546*pi,1.6979496166864871*pi) q[4];
U1q(1.52047158485604*pi,1.744566334089918*pi) q[5];
U1q(1.329494673959*pi,0.9995455688292143*pi) q[6];
U1q(1.84402961699321*pi,1.2374338636430098*pi) q[7];
U1q(0.341585495191213*pi,0.8080920546922972*pi) q[8];
U1q(1.3565977581049*pi,1.472880910986623*pi) q[9];
U1q(3.6203260073695622*pi,1.2067683348575904*pi) q[10];
U1q(1.3212536265394*pi,0.5825485615433523*pi) q[11];
U1q(0.596753100081263*pi,0.3514403582093344*pi) q[12];
U1q(0.477170270148221*pi,0.8141931942385476*pi) q[13];
U1q(3.402059026521505*pi,0.7952933050191175*pi) q[14];
U1q(0.788136330527384*pi,0.459998577432037*pi) q[15];
U1q(3.567254666036165*pi,1.6156782261922231*pi) q[16];
U1q(0.462734905755178*pi,1.774623529815159*pi) q[17];
U1q(0.873903532241235*pi,1.3886838052194177*pi) q[18];
U1q(0.468895434537916*pi,0.6199990216795079*pi) q[19];
U1q(1.32314451161704*pi,1.2998957892565581*pi) q[20];
U1q(3.47675497300953*pi,1.8730418210370123*pi) q[21];
U1q(1.20035285577287*pi,0.14717043395729945*pi) q[22];
U1q(3.175854014305433*pi,1.455558775170041*pi) q[23];
U1q(1.75225819275756*pi,1.6364820154402722*pi) q[24];
U1q(0.449144307152316*pi,0.9068021012681007*pi) q[25];
U1q(1.59117111171869*pi,0.5843958754314733*pi) q[26];
U1q(3.113397069863332*pi,0.7579240494451494*pi) q[27];
U1q(3.192755138252731*pi,1.6369682374671717*pi) q[28];
U1q(0.73408588460791*pi,0.9551707690000395*pi) q[29];
U1q(0.517813259762148*pi,0.23795910170959766*pi) q[30];
U1q(1.24947345320526*pi,0.41286507032514397*pi) q[31];
U1q(1.5472618535685*pi,0.79490257863237*pi) q[32];
U1q(1.40825744041786*pi,0.14090959749583298*pi) q[33];
U1q(1.23697569943375*pi,1.9005150581040384*pi) q[34];
U1q(1.51770076566846*pi,1.3153009415131822*pi) q[35];
U1q(3.475739235069436*pi,1.7636263213361558*pi) q[36];
U1q(0.0713264112572142*pi,1.0736339374841162*pi) q[37];
U1q(3.538620430104023*pi,1.0049866963805645*pi) q[38];
U1q(1.69025358271035*pi,1.5758299873465642*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[16],q[1];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[5],q[8];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[28],q[12];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[14],q[39];
RZZ(0.5*pi) q[38],q[15];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[33],q[21];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[24],q[26];
RZZ(0.5*pi) q[37],q[27];
RZZ(0.5*pi) q[34],q[31];
U1q(0.262741869543658*pi,0.77922068494955*pi) q[0];
U1q(0.815842837929437*pi,0.5193251734937325*pi) q[1];
U1q(1.62363931499043*pi,0.889655295123331*pi) q[2];
U1q(0.35385935577842*pi,1.2096690288584426*pi) q[3];
U1q(1.57420863747571*pi,0.9355886290324968*pi) q[4];
U1q(3.438257377034792*pi,1.014964309776543*pi) q[5];
U1q(1.4816103631485*pi,1.2464332714239434*pi) q[6];
U1q(0.235614560809938*pi,0.31754217116275996*pi) q[7];
U1q(0.432719844519581*pi,0.06209727715959801*pi) q[8];
U1q(3.475830113577045*pi,0.012833753426755301*pi) q[9];
U1q(1.43630639007695*pi,0.7506306185106832*pi) q[10];
U1q(3.850757404620048*pi,0.2364671324083094*pi) q[11];
U1q(0.143811982934183*pi,1.0757348837251133*pi) q[12];
U1q(0.793281168823174*pi,0.29140013170903734*pi) q[13];
U1q(1.4254942821685*pi,1.358177519804685*pi) q[14];
U1q(0.507928089161505*pi,1.5238263782137675*pi) q[15];
U1q(3.211790257901738*pi,0.1901012180940871*pi) q[16];
U1q(0.126073659812986*pi,1.9151444579068988*pi) q[17];
U1q(0.914876000014347*pi,1.120499839942859*pi) q[18];
U1q(0.694928629619949*pi,0.05237814364701876*pi) q[19];
U1q(3.4523749684979013*pi,0.6708647742953371*pi) q[20];
U1q(1.18063933968942*pi,0.06898164220931435*pi) q[21];
U1q(1.54288904263758*pi,1.8674854447579108*pi) q[22];
U1q(1.18331773083411*pi,0.8614941192159491*pi) q[23];
U1q(0.71568966385054*pi,1.4328876603854814*pi) q[24];
U1q(0.233674587540156*pi,1.9522687305245605*pi) q[25];
U1q(0.648556411543237*pi,0.5154278871448135*pi) q[26];
U1q(1.33573373152429*pi,1.307257444143338*pi) q[27];
U1q(1.57715031905627*pi,1.5384959149617288*pi) q[28];
U1q(0.405890076692984*pi,0.26226278110516077*pi) q[29];
U1q(0.163562712069571*pi,1.1673366120122175*pi) q[30];
U1q(3.332884670678896*pi,1.1478453385648493*pi) q[31];
U1q(0.0581590409277257*pi,0.702935522630451*pi) q[32];
U1q(0.400737541886366*pi,1.1015045281991842*pi) q[33];
U1q(1.04375758563294*pi,0.15278086471220576*pi) q[34];
U1q(0.114641279859539*pi,1.8877479241314852*pi) q[35];
U1q(0.432343880221001*pi,1.2291721979587256*pi) q[36];
U1q(0.865369005137155*pi,1.5719116288358759*pi) q[37];
U1q(1.6007309446095*pi,0.8605111592835799*pi) q[38];
U1q(3.424013634565704*pi,1.8301368833938962*pi) q[39];
rz(1.22077931505045*pi) q[0];
rz(3.4806748265062675*pi) q[1];
rz(3.110344704876669*pi) q[2];
rz(0.7903309711415574*pi) q[3];
rz(3.064411370967503*pi) q[4];
rz(0.9850356902234568*pi) q[5];
rz(2.7535667285760566*pi) q[6];
rz(3.68245782883724*pi) q[7];
rz(1.937902722840402*pi) q[8];
rz(1.9871662465732447*pi) q[9];
rz(3.2493693814893168*pi) q[10];
rz(1.7635328675916906*pi) q[11];
rz(2.9242651162748867*pi) q[12];
rz(1.7085998682909627*pi) q[13];
rz(2.641822480195315*pi) q[14];
rz(0.47617362178623246*pi) q[15];
rz(1.809898781905913*pi) q[16];
rz(2.084855542093101*pi) q[17];
rz(0.879500160057141*pi) q[18];
rz(3.9476218563529812*pi) q[19];
rz(1.329135225704663*pi) q[20];
rz(3.931018357790686*pi) q[21];
rz(2.132514555242089*pi) q[22];
rz(1.138505880784051*pi) q[23];
rz(2.5671123396145186*pi) q[24];
rz(0.047731269475439375*pi) q[25];
rz(1.4845721128551865*pi) q[26];
rz(0.692742555856662*pi) q[27];
rz(2.4615040850382712*pi) q[28];
rz(1.7377372188948392*pi) q[29];
rz(0.8326633879877825*pi) q[30];
rz(0.8521546614351507*pi) q[31];
rz(1.297064477369549*pi) q[32];
rz(2.898495471800816*pi) q[33];
rz(1.8472191352877942*pi) q[34];
rz(2.112252075868515*pi) q[35];
rz(2.7708278020412744*pi) q[36];
rz(2.428088371164124*pi) q[37];
rz(1.13948884071642*pi) q[38];
rz(2.1698631166061038*pi) q[39];
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
