OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.316724102888461*pi,1.07158422994519*pi) q[0];
U1q(0.0596228654317132*pi,1.825121596842724*pi) q[1];
U1q(1.11464865012442*pi,1.2935320176560776*pi) q[2];
U1q(0.369579526896794*pi,1.471176278906283*pi) q[3];
U1q(3.496539484287694*pi,1.011714133556434*pi) q[4];
U1q(0.53655743604532*pi,1.255729596191312*pi) q[5];
U1q(0.19655466299963*pi,0.188869194801775*pi) q[6];
U1q(0.580450390780137*pi,1.417577675572949*pi) q[7];
U1q(0.18993348167571*pi,1.55228095011481*pi) q[8];
U1q(0.678459222030188*pi,1.616939563417459*pi) q[9];
U1q(3.162321447961378*pi,1.5015360990370576*pi) q[10];
U1q(0.744873290451618*pi,0.554506585888873*pi) q[11];
U1q(0.661927760431896*pi,0.774912886872833*pi) q[12];
U1q(1.69550437554673*pi,0.6599934396050711*pi) q[13];
U1q(3.380910663550162*pi,1.0215750874522682*pi) q[14];
U1q(1.44150262514209*pi,1.7946793981357394*pi) q[15];
U1q(1.59036025229508*pi,1.3358298590566071*pi) q[16];
U1q(0.789442893367973*pi,1.1669228265557*pi) q[17];
U1q(1.69424328805408*pi,1.146162796454365*pi) q[18];
U1q(0.771665336540327*pi,1.579393219169708*pi) q[19];
U1q(0.316798998162818*pi,0.690224378250552*pi) q[20];
U1q(1.24520745197631*pi,1.9210693145176185*pi) q[21];
U1q(1.54382198679262*pi,0.19004974748613726*pi) q[22];
U1q(0.683972083338644*pi,1.700018466505592*pi) q[23];
U1q(0.462743892989553*pi,1.83868573450934*pi) q[24];
U1q(3.788060910342952*pi,1.0409480408352012*pi) q[25];
U1q(0.0494320938435547*pi,0.197706317760954*pi) q[26];
U1q(1.34480151365175*pi,1.492121772668012*pi) q[27];
U1q(1.79258100710976*pi,1.479779590073743*pi) q[28];
U1q(0.809950396843327*pi,0.277962353938834*pi) q[29];
U1q(3.341024971695476*pi,1.2489181608543864*pi) q[30];
U1q(0.723541004498081*pi,0.286813263769847*pi) q[31];
U1q(1.56145328991582*pi,1.6492663265487686*pi) q[32];
U1q(0.269019470774287*pi,0.227784108061443*pi) q[33];
U1q(0.961042591290183*pi,1.15867861819901*pi) q[34];
U1q(0.570820258118291*pi,0.167854777118837*pi) q[35];
U1q(1.60340981201803*pi,0.5505486549904693*pi) q[36];
U1q(1.50149981929598*pi,1.1078386670961509*pi) q[37];
U1q(0.471018845069146*pi,1.014446186052564*pi) q[38];
U1q(1.71650351630636*pi,1.5003407280046392*pi) q[39];
RZZ(0.5*pi) q[0],q[29];
RZZ(0.5*pi) q[1],q[34];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[6],q[14];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[16],q[30];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[19],q[21];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[35],q[27];
RZZ(0.5*pi) q[38],q[37];
U1q(0.545103553718906*pi,0.67576356080701*pi) q[0];
U1q(0.58411784664367*pi,1.24699138833338*pi) q[1];
U1q(0.906242218028758*pi,1.5668329674545678*pi) q[2];
U1q(0.64906705758143*pi,0.2516230260294601*pi) q[3];
U1q(0.437174778333054*pi,0.689646633098794*pi) q[4];
U1q(0.291129941224909*pi,0.09189166458946008*pi) q[5];
U1q(0.341819352080733*pi,1.8266813377318103*pi) q[6];
U1q(0.262686253509778*pi,0.9551182734063302*pi) q[7];
U1q(0.610069310683126*pi,0.26798243741386996*pi) q[8];
U1q(0.919678613337842*pi,0.06296200801113994*pi) q[9];
U1q(0.577142933887767*pi,1.7699204604319774*pi) q[10];
U1q(0.322409772895404*pi,1.9194303782138*pi) q[11];
U1q(0.412678167314811*pi,0.2750774757699299*pi) q[12];
U1q(0.242898972661645*pi,1.5385266082059412*pi) q[13];
U1q(0.525392540678187*pi,0.03682449440585622*pi) q[14];
U1q(0.345002644281079*pi,1.6017051477525195*pi) q[15];
U1q(0.761985372410769*pi,1.329439767859637*pi) q[16];
U1q(0.253475011644403*pi,0.17078794616126003*pi) q[17];
U1q(0.864309711606556*pi,1.32482690130167*pi) q[18];
U1q(0.0712800505731326*pi,0.9150439447996899*pi) q[19];
U1q(0.420756899584762*pi,0.49905449280894*pi) q[20];
U1q(0.470552836346392*pi,0.28130749136596855*pi) q[21];
U1q(0.853127271928016*pi,0.1723556477403274*pi) q[22];
U1q(0.351182343969784*pi,0.2955540964382699*pi) q[23];
U1q(0.737784017509725*pi,1.86652673444802*pi) q[24];
U1q(0.558528166562172*pi,0.8641635114896014*pi) q[25];
U1q(0.635644236873256*pi,0.21815141161197005*pi) q[26];
U1q(0.351062963018493*pi,0.3669093995749222*pi) q[27];
U1q(0.281185272463889*pi,0.5514483619921631*pi) q[28];
U1q(0.853522901057337*pi,0.81856658937112*pi) q[29];
U1q(0.370063179102767*pi,1.1620047093427965*pi) q[30];
U1q(0.875221567677062*pi,1.3663868245102502*pi) q[31];
U1q(0.306505536110205*pi,1.7905007999521985*pi) q[32];
U1q(0.325441590567383*pi,0.19710011345646983*pi) q[33];
U1q(0.468148478667468*pi,1.9106843339354858*pi) q[34];
U1q(0.757098194559654*pi,1.3637729215652898*pi) q[35];
U1q(0.151368433311521*pi,0.5492673888165496*pi) q[36];
U1q(0.320184375958024*pi,0.2132860293494807*pi) q[37];
U1q(0.630008212960003*pi,0.82884436089925*pi) q[38];
U1q(0.394661359607612*pi,1.8104966978852093*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[2],q[14];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[31];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[10],q[22];
RZZ(0.5*pi) q[11],q[29];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[27],q[39];
RZZ(0.5*pi) q[36],q[34];
U1q(0.340921186158731*pi,0.6553605754287202*pi) q[0];
U1q(0.680097640883474*pi,0.5495749103805698*pi) q[1];
U1q(0.591817292063071*pi,1.4812429746003177*pi) q[2];
U1q(0.463464242279495*pi,1.1393486344295898*pi) q[3];
U1q(0.566896428083024*pi,0.5127387710681841*pi) q[4];
U1q(0.135178792642435*pi,0.49250683640535*pi) q[5];
U1q(0.598014312453424*pi,0.5643470056552804*pi) q[6];
U1q(0.634842025943289*pi,1.1318860614253996*pi) q[7];
U1q(0.702998054195997*pi,1.15143068653401*pi) q[8];
U1q(0.584527656848972*pi,1.8894580711499103*pi) q[9];
U1q(0.378915380840218*pi,0.37501180733495776*pi) q[10];
U1q(0.31649565263477*pi,1.86264101682805*pi) q[11];
U1q(0.292553241417612*pi,1.7875183077754704*pi) q[12];
U1q(0.167668137065762*pi,0.022388969270360715*pi) q[13];
U1q(0.492017189459422*pi,1.0072836965654082*pi) q[14];
U1q(0.351623328478313*pi,0.9207240866229087*pi) q[15];
U1q(0.188545135559377*pi,0.9839662597537773*pi) q[16];
U1q(0.254127969718442*pi,1.07656731986678*pi) q[17];
U1q(0.45270504309825*pi,1.2217937444089948*pi) q[18];
U1q(0.210694225253744*pi,0.08450883787612007*pi) q[19];
U1q(0.474291175010471*pi,0.06316551876181009*pi) q[20];
U1q(0.116497579878501*pi,1.5856261286041784*pi) q[21];
U1q(0.269304583783037*pi,0.5277681083245973*pi) q[22];
U1q(0.645724256720809*pi,1.6358333893895098*pi) q[23];
U1q(0.445511871830402*pi,0.9743831097863098*pi) q[24];
U1q(0.456848771248589*pi,0.0795847771677014*pi) q[25];
U1q(0.645423568687186*pi,1.9793375665026502*pi) q[26];
U1q(0.256056273090592*pi,0.13552153180988213*pi) q[27];
U1q(0.507340086634335*pi,0.12032840350591378*pi) q[28];
U1q(0.661467001421493*pi,0.42553075574068*pi) q[29];
U1q(0.537315949257201*pi,0.14623437563438646*pi) q[30];
U1q(0.926920948724733*pi,0.9701997862792302*pi) q[31];
U1q(0.444759567783519*pi,0.5809653331762084*pi) q[32];
U1q(0.514899166548262*pi,1.9861789579566898*pi) q[33];
U1q(0.23363225646232*pi,1.5169285736276201*pi) q[34];
U1q(0.587549451048101*pi,0.89045423982328*pi) q[35];
U1q(0.275388194965714*pi,0.17609210294712874*pi) q[36];
U1q(0.647908161705816*pi,0.8137004971409505*pi) q[37];
U1q(0.587948326045416*pi,1.79013479634046*pi) q[38];
U1q(0.763103496521508*pi,1.5653033705120492*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[5],q[20];
RZZ(0.5*pi) q[6],q[28];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[35],q[11];
RZZ(0.5*pi) q[12],q[39];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[19],q[14];
RZZ(0.5*pi) q[17],q[15];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[24],q[32];
RZZ(0.5*pi) q[30],q[29];
RZZ(0.5*pi) q[38],q[34];
U1q(0.92546919586896*pi,0.2604356768842804*pi) q[0];
U1q(0.860287428594048*pi,0.71338074799172*pi) q[1];
U1q(0.485322471549957*pi,0.3424695114594778*pi) q[2];
U1q(0.32014826000096*pi,1.4849149643487998*pi) q[3];
U1q(0.314852203717203*pi,0.8490806717365356*pi) q[4];
U1q(0.18816024562317*pi,0.3778175643319299*pi) q[5];
U1q(0.745389269195316*pi,1.20499204034705*pi) q[6];
U1q(0.454661622697442*pi,1.788898789308*pi) q[7];
U1q(0.415374141363471*pi,0.7346585494177003*pi) q[8];
U1q(0.507835927796965*pi,0.4352432932700303*pi) q[9];
U1q(0.81752109906118*pi,0.8403138064166971*pi) q[10];
U1q(0.572728363611297*pi,1.7537221165983592*pi) q[11];
U1q(0.432644930122333*pi,0.3601551862613599*pi) q[12];
U1q(0.716407870543763*pi,1.3023705697005319*pi) q[13];
U1q(0.456990058508489*pi,0.8721870653430281*pi) q[14];
U1q(0.608076689395555*pi,0.020263310098038723*pi) q[15];
U1q(0.676109660141588*pi,0.6169819973426875*pi) q[16];
U1q(0.447276877851412*pi,1.19519055038488*pi) q[17];
U1q(0.240724389908233*pi,0.2969069993025446*pi) q[18];
U1q(0.348931574297755*pi,1.4292215508599995*pi) q[19];
U1q(0.583491253557321*pi,0.6775748257562997*pi) q[20];
U1q(0.825732827226785*pi,0.5443630595089983*pi) q[21];
U1q(0.612191939437821*pi,0.9099636893103371*pi) q[22];
U1q(0.502086051470329*pi,0.6952855719244297*pi) q[23];
U1q(0.640539645199559*pi,0.38621518859994985*pi) q[24];
U1q(0.761198270738602*pi,0.747467194514301*pi) q[25];
U1q(0.669088073195389*pi,0.09510549506313026*pi) q[26];
U1q(0.399660514922266*pi,0.2014455048167827*pi) q[27];
U1q(0.294827165277099*pi,0.7904473405190835*pi) q[28];
U1q(0.26087356791605*pi,1.5430056166002997*pi) q[29];
U1q(0.388350896017005*pi,0.03632916474898629*pi) q[30];
U1q(0.557512466774738*pi,1.9091261729199802*pi) q[31];
U1q(0.569362673800593*pi,1.4662221842297694*pi) q[32];
U1q(0.666823544963589*pi,1.27670295698751*pi) q[33];
U1q(0.504091102411858*pi,0.9103637859194302*pi) q[34];
U1q(0.545215261575307*pi,0.2638365269081202*pi) q[35];
U1q(0.480315586861701*pi,1.9510229274259796*pi) q[36];
U1q(0.464472104981306*pi,0.24576169950288929*pi) q[37];
U1q(0.313566931917242*pi,1.16181755357296*pi) q[38];
U1q(0.42572097854675*pi,1.1310499090332096*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[18],q[1];
RZZ(0.5*pi) q[2],q[28];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[35],q[4];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[6],q[22];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[23],q[8];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[10],q[31];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[14],q[34];
RZZ(0.5*pi) q[15],q[29];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[26],q[20];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[24],q[39];
U1q(0.279179951585273*pi,0.62329131000822*pi) q[0];
U1q(0.491260908140798*pi,1.7991170205374107*pi) q[1];
U1q(0.339522176141001*pi,1.7589417042721784*pi) q[2];
U1q(0.250557285691847*pi,1.1030617618318708*pi) q[3];
U1q(0.499567066398285*pi,0.10353650213403398*pi) q[4];
U1q(0.326683214870418*pi,0.9091397768201297*pi) q[5];
U1q(0.497901461283941*pi,1.2343592921187394*pi) q[6];
U1q(0.251004526426626*pi,0.2224983001840002*pi) q[7];
U1q(0.185733250814238*pi,0.5305138911451497*pi) q[8];
U1q(0.411227043302804*pi,1.9343717902189006*pi) q[9];
U1q(0.330963209697438*pi,0.14382052057882788*pi) q[10];
U1q(0.862454306418798*pi,0.39904078887395045*pi) q[11];
U1q(0.774060428150884*pi,0.13910709076901995*pi) q[12];
U1q(0.596141134592157*pi,0.1783147109133001*pi) q[13];
U1q(0.637640239787659*pi,1.862095543967719*pi) q[14];
U1q(0.40796416342261*pi,0.231625573143039*pi) q[15];
U1q(0.563880457478195*pi,1.2620845088029977*pi) q[16];
U1q(0.153969232520865*pi,0.36301213013005995*pi) q[17];
U1q(0.34936908256427*pi,1.1874054642804248*pi) q[18];
U1q(0.132575239282429*pi,1.1377017782346002*pi) q[19];
U1q(0.180788938525518*pi,1.1877486326239302*pi) q[20];
U1q(0.0577879234197546*pi,1.959200258119088*pi) q[21];
U1q(0.556424775914281*pi,1.8473491738147079*pi) q[22];
U1q(0.161133233606847*pi,1.2654839935337794*pi) q[23];
U1q(0.360232275698389*pi,0.9218873718579204*pi) q[24];
U1q(0.801724688272323*pi,0.7402638050048509*pi) q[25];
U1q(0.799825341471133*pi,1.3568853454777994*pi) q[26];
U1q(0.530940466706895*pi,0.7840133905734827*pi) q[27];
U1q(0.4106613286889*pi,0.5187759702376429*pi) q[28];
U1q(0.116508788476781*pi,0.26456765322289044*pi) q[29];
U1q(0.28114006344388*pi,0.853661119372287*pi) q[30];
U1q(0.399084316481031*pi,1.2844323316548598*pi) q[31];
U1q(0.812077164754599*pi,0.6950075703336687*pi) q[32];
U1q(0.361446909956722*pi,0.23959034872260077*pi) q[33];
U1q(0.50442521346352*pi,0.45422735705748973*pi) q[34];
U1q(0.580985693199188*pi,0.6034073763426004*pi) q[35];
U1q(0.222898846728867*pi,1.4785492263617694*pi) q[36];
U1q(0.400722528086285*pi,1.3502998242862496*pi) q[37];
U1q(0.133830070295671*pi,1.3822555610614602*pi) q[38];
U1q(0.717186080619579*pi,0.4379122035869196*pi) q[39];
RZZ(0.5*pi) q[0],q[34];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[2],q[8];
RZZ(0.5*pi) q[3],q[20];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[26],q[9];
RZZ(0.5*pi) q[10],q[39];
RZZ(0.5*pi) q[11],q[21];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[13],q[15];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[38],q[25];
RZZ(0.5*pi) q[33],q[28];
RZZ(0.5*pi) q[31],q[29];
RZZ(0.5*pi) q[32],q[36];
U1q(0.746255463688159*pi,0.1448732225409195*pi) q[0];
U1q(0.425453575333665*pi,1.5417215909392006*pi) q[1];
U1q(0.322730435918561*pi,1.2034395394723791*pi) q[2];
U1q(0.793454272706843*pi,0.8895744929117004*pi) q[3];
U1q(0.128314111360878*pi,1.1103171257354347*pi) q[4];
U1q(0.565285804903254*pi,0.9356714093467993*pi) q[5];
U1q(0.357092700286514*pi,1.2812656779678004*pi) q[6];
U1q(0.912806101802796*pi,1.0293107070447007*pi) q[7];
U1q(0.694770169000028*pi,1.6425263161809998*pi) q[8];
U1q(0.236204496396582*pi,1.4013873648413995*pi) q[9];
U1q(0.305384874056886*pi,1.027449268426757*pi) q[10];
U1q(0.749267020145808*pi,0.4633055046524994*pi) q[11];
U1q(0.816966343229037*pi,1.7615330496282002*pi) q[12];
U1q(0.494228788544271*pi,1.8798913367667698*pi) q[13];
U1q(0.315299565163795*pi,1.9140066352824583*pi) q[14];
U1q(0.455718394980162*pi,1.6991416521165394*pi) q[15];
U1q(0.570219468710473*pi,1.1723842895594068*pi) q[16];
U1q(0.373775120738865*pi,1.7257886308851997*pi) q[17];
U1q(0.278435579677329*pi,1.5183153900596036*pi) q[18];
U1q(0.794269672286235*pi,1.4554478014308003*pi) q[19];
U1q(0.334327065978405*pi,0.08418897480791987*pi) q[20];
U1q(0.168510880310619*pi,0.7146033255863191*pi) q[21];
U1q(0.882955402103143*pi,0.6679853277031178*pi) q[22];
U1q(0.280539325429516*pi,0.5352268841036008*pi) q[23];
U1q(0.427627775212523*pi,0.8861351234845998*pi) q[24];
U1q(0.230614988707688*pi,1.2075292195114606*pi) q[25];
U1q(0.107687491360856*pi,0.028104273811800695*pi) q[26];
U1q(0.506789463864398*pi,0.3764794518565129*pi) q[27];
U1q(0.693439317016805*pi,0.6048474612193431*pi) q[28];
U1q(0.20831068186741*pi,1.0277904646824005*pi) q[29];
U1q(0.655952362194831*pi,0.48408941226088764*pi) q[30];
U1q(0.591828623310395*pi,1.1795881265118293*pi) q[31];
U1q(0.368999681284453*pi,1.751771804677567*pi) q[32];
U1q(0.546265289190897*pi,0.21994188255919944*pi) q[33];
U1q(0.406456761374049*pi,1.9653359672191009*pi) q[34];
U1q(0.681682824996894*pi,1.3858455985999996*pi) q[35];
U1q(0.698512178211079*pi,1.2796978413808695*pi) q[36];
U1q(0.523318334059224*pi,1.4812835313585495*pi) q[37];
U1q(0.280400400720535*pi,1.8341110249537298*pi) q[38];
U1q(0.339786643621694*pi,0.13924369944365012*pi) q[39];
RZZ(0.5*pi) q[0],q[10];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[38],q[3];
RZZ(0.5*pi) q[4],q[36];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[7],q[29];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[35],q[15];
RZZ(0.5*pi) q[17],q[37];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[24],q[34];
RZZ(0.5*pi) q[33],q[25];
RZZ(0.5*pi) q[27],q[31];
RZZ(0.5*pi) q[32],q[28];
U1q(0.724308123021286*pi,1.0587445056559996*pi) q[0];
U1q(0.274568844930167*pi,0.44578233813079926*pi) q[1];
U1q(0.303424455022273*pi,0.17937697870147673*pi) q[2];
U1q(0.653671143800693*pi,1.709154461692*pi) q[3];
U1q(0.648851693431973*pi,0.2876360968650342*pi) q[4];
U1q(0.885802288596602*pi,1.4216262020374*pi) q[5];
U1q(0.259732009277205*pi,0.2890105191555996*pi) q[6];
U1q(0.848704147987022*pi,0.9500841541717993*pi) q[7];
U1q(0.889474535749114*pi,1.3069080889304008*pi) q[8];
U1q(0.41835376357855*pi,1.8858370522265009*pi) q[9];
U1q(0.295320840579594*pi,1.9520594862742584*pi) q[10];
U1q(0.82921537933728*pi,0.7877764951363009*pi) q[11];
U1q(0.33519432572974*pi,0.6135245083350007*pi) q[12];
U1q(0.295605552374998*pi,1.684425826642471*pi) q[13];
U1q(0.355174271905931*pi,1.2968473476714184*pi) q[14];
U1q(0.24839868911473*pi,1.8922162711698398*pi) q[15];
U1q(0.530161970114236*pi,0.7375941196808071*pi) q[16];
U1q(0.363930251354815*pi,0.26609263081289924*pi) q[17];
U1q(0.788128129976424*pi,1.1216403642916646*pi) q[18];
U1q(0.214427450553325*pi,0.35963503867260016*pi) q[19];
U1q(0.564618996465304*pi,0.5366001107847094*pi) q[20];
U1q(0.226841626243405*pi,0.42861255613181903*pi) q[21];
U1q(0.201395027204781*pi,1.497345024230837*pi) q[22];
U1q(0.921828104844492*pi,0.9155418915691005*pi) q[23];
U1q(0.552329613720675*pi,1.1832774201419998*pi) q[24];
U1q(0.916016442394532*pi,1.6107625834596*pi) q[25];
U1q(0.278049628422377*pi,1.7353887555466994*pi) q[26];
U1q(0.303486868993693*pi,1.4713728984312127*pi) q[27];
U1q(0.440127624052361*pi,0.9756122811743442*pi) q[28];
U1q(0.525851761225247*pi,1.8813738871615993*pi) q[29];
U1q(0.533917291938441*pi,0.4855180094955873*pi) q[30];
U1q(0.581722349471942*pi,1.9088263244223*pi) q[31];
U1q(0.528791764066522*pi,0.8579912120474695*pi) q[32];
U1q(0.784895382090261*pi,0.7356085269620003*pi) q[33];
U1q(0.530510727171076*pi,0.04027953056169942*pi) q[34];
U1q(0.249303581459601*pi,1.7041538210020999*pi) q[35];
U1q(0.543616119959424*pi,0.5817064810332706*pi) q[36];
U1q(0.600423606406007*pi,1.4057736992238503*pi) q[37];
U1q(0.603306153707278*pi,0.8168431334066995*pi) q[38];
U1q(0.27855566061331*pi,0.33625951189273984*pi) q[39];
rz(3.7947109987914995*pi) q[0];
rz(0.6840719527579004*pi) q[1];
rz(1.348909490880125*pi) q[2];
rz(2.4216827815066004*pi) q[3];
rz(0.2435920919597656*pi) q[4];
rz(1.9323028952910999*pi) q[5];
rz(1.4855525688707*pi) q[6];
rz(2.6276817362198006*pi) q[7];
rz(3.3732029537593995*pi) q[8];
rz(3.4797580878827006*pi) q[9];
rz(0.7556782911330409*pi) q[10];
rz(0.39729839231319986*pi) q[11];
rz(3.2661761831166007*pi) q[12];
rz(1.2403597484080287*pi) q[13];
rz(3.652729200291832*pi) q[14];
rz(1.7275429110274594*pi) q[15];
rz(0.021880182251393876*pi) q[16];
rz(3.2737440937074*pi) q[17];
rz(1.416442858610436*pi) q[18];
rz(1.8154107973201015*pi) q[19];
rz(1.1839355892444594*pi) q[20];
rz(0.06084279375478019*pi) q[21];
rz(3.7783057982843626*pi) q[22];
rz(1.6909450107607*pi) q[23];
rz(3.9456390244243007*pi) q[24];
rz(0.8790818199413977*pi) q[25];
rz(2.6564953313135007*pi) q[26];
rz(2.139765493649488*pi) q[27];
rz(0.33327843117655576*pi) q[28];
rz(1.2959979211677997*pi) q[29];
rz(1.6647483605456124*pi) q[30];
rz(0.6972583470651994*pi) q[31];
rz(0.3600585422008322*pi) q[32];
rz(0.4161171030850994*pi) q[33];
rz(0.7419192590314001*pi) q[34];
rz(1.4761853284487998*pi) q[35];
rz(3.255206199549031*pi) q[36];
rz(1.454904939138249*pi) q[37];
rz(3.2366101973422*pi) q[38];
rz(3.43512941326056*pi) q[39];
U1q(1.72430812302129*pi,1.853455504447586*pi) q[0];
U1q(1.27456884493017*pi,0.129854290888744*pi) q[1];
U1q(0.303424455022273*pi,0.528286469581639*pi) q[2];
U1q(0.653671143800693*pi,1.13083724319858*pi) q[3];
U1q(1.64885169343197*pi,1.53122818882486*pi) q[4];
U1q(3.885802288596602*pi,0.353929097328481*pi) q[5];
U1q(0.259732009277205*pi,0.774563088026309*pi) q[6];
U1q(0.848704147987022*pi,0.577765890391606*pi) q[7];
U1q(3.889474535749114*pi,1.680111042689765*pi) q[8];
U1q(0.41835376357855*pi,0.365595140109248*pi) q[9];
U1q(1.29532084057959*pi,1.707737777407299*pi) q[10];
U1q(0.82921537933728*pi,0.185074887449531*pi) q[11];
U1q(0.33519432572974*pi,0.879700691451572*pi) q[12];
U1q(1.295605552375*pi,1.9247855750504843*pi) q[13];
U1q(0.355174271905931*pi,1.9495765479632883*pi) q[14];
U1q(0.24839868911473*pi,0.619759182197302*pi) q[15];
U1q(0.530161970114236*pi,1.759474301932201*pi) q[16];
U1q(1.36393025135482*pi,0.539836724520334*pi) q[17];
U1q(1.78812812997642*pi,1.538083222902139*pi) q[18];
U1q(1.21442745055333*pi,1.175045835992746*pi) q[19];
U1q(0.564618996465304*pi,0.720535700029171*pi) q[20];
U1q(1.2268416262434*pi,1.4894553498866339*pi) q[21];
U1q(0.201395027204781*pi,0.275650822515248*pi) q[22];
U1q(1.92182810484449*pi,1.606486902329793*pi) q[23];
U1q(1.55232961372068*pi,0.128916444566296*pi) q[24];
U1q(1.91601644239453*pi,1.489844403400945*pi) q[25];
U1q(0.278049628422377*pi,1.3918840868602729*pi) q[26];
U1q(1.30348686899369*pi,0.611138392080682*pi) q[27];
U1q(1.44012762405236*pi,0.308890712350953*pi) q[28];
U1q(1.52585176122525*pi,0.17737180832938*pi) q[29];
U1q(1.53391729193844*pi,1.150266370041154*pi) q[30];
U1q(0.581722349471942*pi,1.606084671487542*pi) q[31];
U1q(1.52879176406652*pi,0.218049754248279*pi) q[32];
U1q(0.784895382090261*pi,0.151725630047106*pi) q[33];
U1q(0.530510727171076*pi,1.782198789593075*pi) q[34];
U1q(1.2493035814596*pi,0.180339149450943*pi) q[35];
U1q(0.543616119959424*pi,0.836912680582385*pi) q[36];
U1q(0.600423606406007*pi,1.860678638362152*pi) q[37];
U1q(0.603306153707278*pi,1.053453330748866*pi) q[38];
U1q(0.27855566061331*pi,0.771388925153318*pi) q[39];
RZZ(0.5*pi) q[0],q[10];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[38],q[3];
RZZ(0.5*pi) q[4],q[36];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[7],q[29];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[35],q[15];
RZZ(0.5*pi) q[17],q[37];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[24],q[34];
RZZ(0.5*pi) q[33],q[25];
RZZ(0.5*pi) q[27],q[31];
RZZ(0.5*pi) q[32],q[28];
U1q(1.74625546368816*pi,1.7673267875627103*pi) q[0];
U1q(3.5745464246663348*pi,0.03391503808043583*pi) q[1];
U1q(1.32273043591856*pi,0.55234903035251*pi) q[2];
U1q(3.793454272706844*pi,1.3112572744183502*pi) q[3];
U1q(1.12831411136088*pi,0.7085471599545335*pi) q[4];
U1q(1.56528580490325*pi,1.8398838900190109*pi) q[5];
U1q(1.35709270028651*pi,0.7668182468384801*pi) q[6];
U1q(1.9128061018028*pi,0.6569924432645702*pi) q[7];
U1q(3.305229830999972*pi,0.34449281543918975*pi) q[8];
U1q(3.236204496396582*pi,1.88114545272418*pi) q[9];
U1q(3.694615125943114*pi,0.6323479952548539*pi) q[10];
U1q(0.749267020145808*pi,0.8606038969657399*pi) q[11];
U1q(1.81696634322904*pi,1.027709232744804*pi) q[12];
U1q(3.505771211455729*pi,0.7293200649261757*pi) q[13];
U1q(1.3152995651638*pi,1.566735835574327*pi) q[14];
U1q(1.45571839498016*pi,0.4266845631440399*pi) q[15];
U1q(0.570219468710473*pi,1.194264471810772*pi) q[16];
U1q(1.37377512073887*pi,1.0801407244480015*pi) q[17];
U1q(3.721564420322672*pi,1.141408197134217*pi) q[18];
U1q(3.205730327713765*pi,0.07923307323458073*pi) q[19];
U1q(1.33432706597841*pi,1.268124564052382*pi) q[20];
U1q(1.16851088031062*pi,0.2034645804322126*pi) q[21];
U1q(0.882955402103143*pi,1.4462911259875089*pi) q[22];
U1q(3.719460674570484*pi,1.9868019097953749*pi) q[23];
U1q(1.42762777521252*pi,0.4260587412237562*pi) q[24];
U1q(3.769385011292312*pi,0.8930777673490449*pi) q[25];
U1q(0.107687491360856*pi,1.6845996051253498*pi) q[26];
U1q(1.5067894638644*pi,1.7060318386554183*pi) q[27];
U1q(1.69343931701681*pi,1.679655532305969*pi) q[28];
U1q(3.79168931813259*pi,0.03095523080851706*pi) q[29];
U1q(3.344047637805169*pi,1.1516949672758225*pi) q[30];
U1q(0.591828623310395*pi,0.876846473577048*pi) q[31];
U1q(3.631000318715547*pi,1.3242691616181888*pi) q[32];
U1q(3.546265289190897*pi,1.636058985644353*pi) q[33];
U1q(1.40645676137405*pi,1.70725522625045*pi) q[34];
U1q(1.68168282499689*pi,0.4986473718530643*pi) q[35];
U1q(1.69851217821108*pi,1.53490404092995*pi) q[36];
U1q(0.523318334059224*pi,1.9361884704968202*pi) q[37];
U1q(0.280400400720535*pi,1.0707212222958802*pi) q[38];
U1q(0.339786643621694*pi,1.574373112704254*pi) q[39];
RZZ(0.5*pi) q[0],q[34];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[2],q[8];
RZZ(0.5*pi) q[3],q[20];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[26],q[9];
RZZ(0.5*pi) q[10],q[39];
RZZ(0.5*pi) q[11],q[21];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[13],q[15];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[38],q[25];
RZZ(0.5*pi) q[33],q[28];
RZZ(0.5*pi) q[31],q[29];
RZZ(0.5*pi) q[32],q[36];
U1q(0.279179951585273*pi,0.2457448750300102*pi) q[0];
U1q(1.4912609081408*pi,0.7765196084821837*pi) q[1];
U1q(3.339522176141001*pi,0.996846865552703*pi) q[2];
U1q(3.7494427143081532*pi,1.097770005498207*pi) q[3];
U1q(0.499567066398285*pi,1.7017665363531695*pi) q[4];
U1q(0.326683214870418*pi,0.8133522574923*pi) q[5];
U1q(1.49790146128394*pi,0.8137246326875422*pi) q[6];
U1q(3.748995473573374*pi,1.463804850125309*pi) q[7];
U1q(3.814266749185762*pi,0.45650524047500984*pi) q[8];
U1q(1.4112270433028*pi,1.3481610273467357*pi) q[9];
U1q(3.6690367903025622*pi,1.5159767431027542*pi) q[10];
U1q(3.862454306418799*pi,1.7963391811871796*pi) q[11];
U1q(3.225939571849116*pi,0.650135191603995*pi) q[12];
U1q(3.5961411345921572*pi,0.4308966907796099*pi) q[13];
U1q(1.63764023978766*pi,0.6186469268890737*pi) q[14];
U1q(1.40796416342261*pi,0.8942006421175923*pi) q[15];
U1q(1.5638804574782*pi,0.28396469105439004*pi) q[16];
U1q(0.153969232520865*pi,0.7173642236928255*pi) q[17];
U1q(3.6506309174357288*pi,1.472318122913397*pi) q[18];
U1q(3.86742476071757*pi,1.3969790964308006*pi) q[19];
U1q(1.18078893852552*pi,1.1645649062363697*pi) q[20];
U1q(0.0577879234197546*pi,0.4480615129650225*pi) q[21];
U1q(1.55642477591428*pi,0.6256549720991*pi) q[22];
U1q(3.161133233606846*pi,0.25654480036515914*pi) q[23];
U1q(0.360232275698389*pi,0.4618109895971201*pi) q[24];
U1q(1.80172468827232*pi,0.36034318185564995*pi) q[25];
U1q(0.799825341471133*pi,1.01338067679134*pi) q[26];
U1q(0.530940466706895*pi,0.11356577737241613*pi) q[27];
U1q(1.4106613286889*pi,1.593584041324256*pi) q[28];
U1q(1.11650878847678*pi,1.7941780422680573*pi) q[29];
U1q(3.71885993655612*pi,0.7821232601644326*pi) q[30];
U1q(1.39908431648103*pi,0.9816906787200801*pi) q[31];
U1q(3.812077164754599*pi,0.38103339596203023*pi) q[32];
U1q(3.361446909956722*pi,1.61641051948101*pi) q[33];
U1q(3.495574786536479*pi,0.2183638364120124*pi) q[34];
U1q(0.580985693199188*pi,1.7162091495956673*pi) q[35];
U1q(3.222898846728867*pi,0.33605265594908024*pi) q[36];
U1q(3.400722528086285*pi,0.80520476342455*pi) q[37];
U1q(0.133830070295671*pi,0.6188657584036101*pi) q[38];
U1q(0.717186080619579*pi,1.8730416168475301*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[18],q[1];
RZZ(0.5*pi) q[2],q[28];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[35],q[4];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[6],q[22];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[23],q[8];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[10],q[31];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[14],q[34];
RZZ(0.5*pi) q[15],q[29];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[26],q[20];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[24],q[39];
U1q(3.9254691958689603*pi,1.8828892419060708*pi) q[0];
U1q(0.860287428594048*pi,1.6907833359364897*pi) q[1];
U1q(1.48532247154996*pi,1.580374672740013*pi) q[2];
U1q(3.679851739999039*pi,0.7159168029812761*pi) q[3];
U1q(1.3148522037172*pi,1.4473107059556787*pi) q[4];
U1q(0.18816024562317*pi,0.2820300450041002*pi) q[5];
U1q(1.74538926919532*pi,0.7843573809158622*pi) q[6];
U1q(3.545338377302558*pi,1.8974043610013063*pi) q[7];
U1q(3.584625858636529*pi,0.2523605822024597*pi) q[8];
U1q(1.50783592779697*pi,0.8490325303978761*pi) q[9];
U1q(1.81752109906118*pi,0.8194834572648793*pi) q[10];
U1q(1.5727283636113*pi,0.4416578534627771*pi) q[11];
U1q(3.567355069877667*pi,0.4290870961116551*pi) q[12];
U1q(0.716407870543763*pi,0.5549525495668299*pi) q[13];
U1q(1.45699005850849*pi,1.6287384482643739*pi) q[14];
U1q(0.608076689395555*pi,1.6828383790725923*pi) q[15];
U1q(3.323890339858412*pi,1.929067202514691*pi) q[16];
U1q(1.44727687785141*pi,1.5495426439476425*pi) q[17];
U1q(1.24072438990823*pi,1.362816587891274*pi) q[18];
U1q(1.34893157429775*pi,0.1054593238053485*pi) q[19];
U1q(1.58349125355732*pi,1.6543910993687376*pi) q[20];
U1q(0.825732827226785*pi,1.033224314354933*pi) q[21];
U1q(3.387808060562179*pi,0.5630404566034715*pi) q[22];
U1q(3.50208605147033*pi,1.6863463787558153*pi) q[23];
U1q(1.64053964519956*pi,0.9261388063391505*pi) q[24];
U1q(3.761198270738603*pi,0.3675465713650903*pi) q[25];
U1q(1.66908807319539*pi,0.75160082637667*pi) q[26];
U1q(0.399660514922266*pi,0.5309978916157212*pi) q[27];
U1q(3.7051728347229*pi,0.32191267104279886*pi) q[28];
U1q(0.26087356791605*pi,0.07261600564546516*pi) q[29];
U1q(1.38835089601701*pi,0.5994552147877523*pi) q[30];
U1q(3.442487533225262*pi,0.3569968374549579*pi) q[31];
U1q(0.569362673800593*pi,0.15224800985814824*pi) q[32];
U1q(0.666823544963589*pi,0.6535231277459372*pi) q[33];
U1q(3.495908897588142*pi,1.7622274075500803*pi) q[34];
U1q(1.54521526157531*pi,0.37663830016119126*pi) q[35];
U1q(1.4803155868617*pi,0.8085263570133132*pi) q[36];
U1q(3.535527895018693*pi,0.909742888207929*pi) q[37];
U1q(0.313566931917242*pi,0.3984277509151104*pi) q[38];
U1q(0.42572097854675*pi,1.56617932229382*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[5],q[20];
RZZ(0.5*pi) q[6],q[28];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[35],q[11];
RZZ(0.5*pi) q[12],q[39];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[19],q[14];
RZZ(0.5*pi) q[17],q[15];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[24],q[32];
RZZ(0.5*pi) q[30],q[29];
RZZ(0.5*pi) q[38],q[34];
U1q(3.659078813841269*pi,1.4879643433616252*pi) q[0];
U1q(3.680097640883475*pi,1.5269774983253437*pi) q[1];
U1q(1.59181729206307*pi,0.4416012095991686*pi) q[2];
U1q(3.5365357577205048*pi,0.061483132900486126*pi) q[3];
U1q(3.433103571916976*pi,1.7836526066240261*pi) q[4];
U1q(1.13517879264243*pi,0.39671931707751984*pi) q[5];
U1q(3.598014312453423*pi,0.4250024156076371*pi) q[6];
U1q(1.63484202594329*pi,0.5544170888838906*pi) q[7];
U1q(3.297001945804003*pi,0.8355884450861502*pi) q[8];
U1q(1.58452765684897*pi,1.3948177525179952*pi) q[9];
U1q(1.37891538084022*pi,1.3541814581831395*pi) q[10];
U1q(1.31649565263477*pi,0.5505767536924671*pi) q[11];
U1q(3.707446758582387*pi,1.001723974597542*pi) q[12];
U1q(3.1676681370657622*pi,0.2749709491366601*pi) q[13];
U1q(1.49201718945942*pi,1.4936418170419858*pi) q[14];
U1q(0.351623328478313*pi,0.5832991555974623*pi) q[15];
U1q(1.18854513555938*pi,1.5620829401036058*pi) q[16];
U1q(1.25412796971844*pi,1.6681658744657453*pi) q[17];
U1q(1.45270504309825*pi,0.287703332997721*pi) q[18];
U1q(1.21069422525374*pi,0.7607466108214185*pi) q[19];
U1q(3.525708824989529*pi,1.2688004063632237*pi) q[20];
U1q(0.116497579878501*pi,0.07448738345011385*pi) q[21];
U1q(3.269304583783037*pi,0.945236037589205*pi) q[22];
U1q(3.354275743279191*pi,0.7457985612907315*pi) q[23];
U1q(3.554488128169598*pi,1.337970885152802*pi) q[24];
U1q(3.543151228751411*pi,1.0354289887116923*pi) q[25];
U1q(1.64542356868719*pi,1.8673687549371527*pi) q[26];
U1q(1.25605627309059*pi,1.4650739186088162*pi) q[27];
U1q(1.50734008663434*pi,0.9920316080559601*pi) q[28];
U1q(0.661467001421493*pi,1.955141144785847*pi) q[29];
U1q(1.5373159492572*pi,1.7093604256731423*pi) q[30];
U1q(1.92692094872473*pi,1.2959232240957057*pi) q[31];
U1q(0.444759567783519*pi,1.266991158804558*pi) q[32];
U1q(0.514899166548262*pi,0.3629991287151173*pi) q[33];
U1q(3.76636774353768*pi,1.1556626198418853*pi) q[34];
U1q(3.412450548951899*pi,0.7500205872460426*pi) q[35];
U1q(3.724611805034285*pi,0.5834571814921623*pi) q[36];
U1q(3.352091838294185*pi,0.3418040905698687*pi) q[37];
U1q(0.587948326045416*pi,1.0267449936826099*pi) q[38];
U1q(1.76310349652151*pi,0.0004327837726600947*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[2],q[14];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[31];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[10],q[22];
RZZ(0.5*pi) q[11],q[29];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[27],q[39];
RZZ(0.5*pi) q[36],q[34];
U1q(3.545103553718906*pi,1.4675613579833398*pi) q[0];
U1q(1.58411784664367*pi,1.8295610203725392*pi) q[1];
U1q(0.906242218028758*pi,1.527191202453409*pi) q[2];
U1q(1.64906705758143*pi,0.9492087413006258*pi) q[3];
U1q(3.562825221666945*pi,1.6067447445934242*pi) q[4];
U1q(1.29112994122491*pi,1.7973344888934095*pi) q[5];
U1q(0.341819352080733*pi,0.687336747684177*pi) q[6];
U1q(0.262686253509778*pi,0.37764930086481474*pi) q[7];
U1q(3.389930689316874*pi,0.7190366942062898*pi) q[8];
U1q(1.91967861333784*pi,1.568321689379225*pi) q[9];
U1q(3.422857066112233*pi,0.959272805086119*pi) q[10];
U1q(3.677590227104596*pi,1.4937873923067144*pi) q[11];
U1q(3.412678167314811*pi,0.5141648066030822*pi) q[12];
U1q(3.757101027338355*pi,0.758833310201072*pi) q[13];
U1q(0.525392540678187*pi,1.523182614882436*pi) q[14];
U1q(1.34500264428108*pi,0.26428021672707214*pi) q[15];
U1q(3.761985372410769*pi,0.9075564482094656*pi) q[16];
U1q(0.253475011644403*pi,1.762386500760233*pi) q[17];
U1q(1.86430971160656*pi,1.1846701761050458*pi) q[18];
U1q(3.9287199494268688*pi,1.930211503897855*pi) q[19];
U1q(3.420756899584762*pi,0.8329114323161049*pi) q[20];
U1q(1.47055283634639*pi,1.770168746211903*pi) q[21];
U1q(0.853127271928016*pi,0.5898235770049345*pi) q[22];
U1q(3.6488176560302152*pi,1.0860778542419802*pi) q[23];
U1q(3.2622159824902752*pi,0.445827260491082*pi) q[24];
U1q(1.55852816656217*pi,0.2508502543897997*pi) q[25];
U1q(1.63564423687326*pi,1.1061826000464725*pi) q[26];
U1q(3.648937036981507*pi,1.2336860508437755*pi) q[27];
U1q(0.281185272463889*pi,0.4231515665422192*pi) q[28];
U1q(1.85352290105734*pi,0.34817697841628714*pi) q[29];
U1q(3.629936820897232*pi,1.6935900919647384*pi) q[30];
U1q(0.875221567677062*pi,0.6921102623267257*pi) q[31];
U1q(1.3065055361102*pi,1.4765266255805383*pi) q[32];
U1q(1.32544159056738*pi,0.5739202842148972*pi) q[33];
U1q(3.531851521332531*pi,1.7619068595340244*pi) q[34];
U1q(1.75709819455965*pi,0.2767019055040283*pi) q[35];
U1q(3.151368433311521*pi,0.2102818956227379*pi) q[36];
U1q(1.32018437595802*pi,0.9422185583613345*pi) q[37];
U1q(1.63000821296*pi,1.0654545582414103*pi) q[38];
U1q(3.605338640392388*pi,1.755239456399499*pi) q[39];
RZZ(0.5*pi) q[0],q[29];
RZZ(0.5*pi) q[1],q[34];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[6],q[14];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[16],q[30];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[19],q[21];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[35],q[27];
RZZ(0.5*pi) q[38],q[37];
U1q(0.316724102888461*pi,1.8633820271215198*pi) q[0];
U1q(0.0596228654317132*pi,0.40769122888188924*pi) q[1];
U1q(0.114648650124422*pi,1.2538902526549185*pi) q[2];
U1q(0.369579526896794*pi,1.1687619941774567*pi) q[3];
U1q(3.496539484287694*pi,0.2846772441357839*pi) q[4];
U1q(0.53655743604532*pi,0.9611724204952594*pi) q[5];
U1q(0.19655466299963*pi,0.04952460475413645*pi) q[6];
U1q(0.580450390780137*pi,0.8401087030314449*pi) q[7];
U1q(1.18993348167571*pi,0.43473818150535504*pi) q[8];
U1q(1.67845922203019*pi,0.014344133972906459*pi) q[9];
U1q(3.162321447961378*pi,0.22765716648103096*pi) q[10];
U1q(1.74487329045162*pi,1.8587111846316358*pi) q[11];
U1q(0.661927760431896*pi,0.014000217705981033*pi) q[12];
U1q(1.69550437554673*pi,0.6373664788019457*pi) q[13];
U1q(0.380910663550162*pi,1.507933207928846*pi) q[14];
U1q(3.44150262514209*pi,1.0713059663438527*pi) q[15];
U1q(3.590360252295082*pi,1.9011663570124977*pi) q[16];
U1q(0.789442893367973*pi,0.7585213811546739*pi) q[17];
U1q(0.694243288054076*pi,0.00600607125774566*pi) q[18];
U1q(1.77166533654033*pi,1.26586222952783*pi) q[19];
U1q(0.316798998162818*pi,1.024081317757715*pi) q[20];
U1q(1.24520745197631*pi,1.1304069230602298*pi) q[21];
U1q(0.543821986792623*pi,0.6075176767507449*pi) q[22];
U1q(1.68397208333864*pi,0.6816134841746546*pi) q[23];
U1q(1.46274389298955*pi,0.4736682604297682*pi) q[24];
U1q(0.788060910342951*pi,0.42763478373539976*pi) q[25];
U1q(3.0494320938435537*pi,0.12662769389748973*pi) q[26];
U1q(1.34480151365175*pi,1.1084736777506927*pi) q[27];
U1q(0.79258100710976*pi,1.3514827946237897*pi) q[28];
U1q(3.809950396843327*pi,0.888781213848576*pi) q[29];
U1q(3.341024971695476*pi,1.6066766404531485*pi) q[30];
U1q(0.723541004498081*pi,1.612536701586326*pi) q[31];
U1q(1.56145328991582*pi,0.6177610989839684*pi) q[32];
U1q(1.26901947077429*pi,0.5432362896099279*pi) q[33];
U1q(1.96104259129018*pi,1.5139125752705023*pi) q[34];
U1q(0.570820258118291*pi,1.0807837610575683*pi) q[35];
U1q(0.603409812018031*pi,1.2115631617966587*pi) q[36];
U1q(0.501499819295976*pi,1.836771196108005*pi) q[37];
U1q(1.47101884506915*pi,0.8798527330880974*pi) q[38];
U1q(1.71650351630636*pi,1.065395426280062*pi) q[39];
rz(2.13661797287848*pi) q[0];
rz(3.5923087711181108*pi) q[1];
rz(2.7461097473450815*pi) q[2];
rz(0.8312380058225433*pi) q[3];
rz(3.715322755864216*pi) q[4];
rz(3.0388275795047406*pi) q[5];
rz(3.9504753952458636*pi) q[6];
rz(3.159891296968555*pi) q[7];
rz(3.565261818494645*pi) q[8];
rz(1.9856558660270935*pi) q[9];
rz(3.772342833518969*pi) q[10];
rz(2.1412888153683642*pi) q[11];
rz(1.985999782294019*pi) q[12];
rz(3.3626335211980543*pi) q[13];
rz(2.492066792071154*pi) q[14];
rz(2.9286940336561473*pi) q[15];
rz(0.0988336429875023*pi) q[16];
rz(3.241478618845326*pi) q[17];
rz(1.9939939287422543*pi) q[18];
rz(0.7341377704721701*pi) q[19];
rz(2.975918682242285*pi) q[20];
rz(2.86959307693977*pi) q[21];
rz(3.392482323249255*pi) q[22];
rz(1.3183865158253454*pi) q[23];
rz(1.5263317395702318*pi) q[24];
rz(1.5723652162646002*pi) q[25];
rz(3.8733723061025103*pi) q[26];
rz(2.891526322249307*pi) q[27];
rz(0.6485172053762103*pi) q[28];
rz(3.111218786151424*pi) q[29];
rz(0.3933233595468515*pi) q[30];
rz(2.387463298413674*pi) q[31];
rz(3.3822389010160316*pi) q[32];
rz(3.456763710390072*pi) q[33];
rz(0.4860874247294978*pi) q[34];
rz(2.9192162389424317*pi) q[35];
rz(2.7884368382033413*pi) q[36];
rz(2.163228803891995*pi) q[37];
rz(3.1201472669119026*pi) q[38];
rz(2.934604573719938*pi) q[39];
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