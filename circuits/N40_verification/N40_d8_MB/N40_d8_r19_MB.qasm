OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.886525246861671*pi,1.89536995208514*pi) q[0];
U1q(1.55221594383361*pi,0.03809192162281729*pi) q[1];
U1q(1.62916162849682*pi,1.2697710058649045*pi) q[2];
U1q(0.437765407131709*pi,0.87054036370298*pi) q[3];
U1q(1.69821388763059*pi,0.4240327940084657*pi) q[4];
U1q(0.116545847646746*pi,1.3053812358988*pi) q[5];
U1q(1.91147637758417*pi,0.00850836716876105*pi) q[6];
U1q(0.907465179521111*pi,0.0924872968073041*pi) q[7];
U1q(1.3868246501595*pi,1.0067056456491459*pi) q[8];
U1q(0.262637591063151*pi,0.658055916505688*pi) q[9];
U1q(1.70683973765579*pi,1.8147316545273777*pi) q[10];
U1q(0.427415525710959*pi,0.93563732380213*pi) q[11];
U1q(0.152625058892149*pi,0.228780201390495*pi) q[12];
U1q(1.81219473895942*pi,1.021595248053929*pi) q[13];
U1q(1.70113998675811*pi,1.245156799008414*pi) q[14];
U1q(1.2885933677392*pi,0.01739570337909319*pi) q[15];
U1q(0.337772524803366*pi,0.0277085930117382*pi) q[16];
U1q(1.43085337843004*pi,0.6228755806260291*pi) q[17];
U1q(0.124735182251578*pi,0.376851009563947*pi) q[18];
U1q(1.75114559784186*pi,0.938505462773883*pi) q[19];
U1q(1.62239199193577*pi,1.278767531890936*pi) q[20];
U1q(1.12592804450257*pi,1.4480137280533925*pi) q[21];
U1q(1.34541302110324*pi,0.6138634883989008*pi) q[22];
U1q(0.483652530379463*pi,0.982923218336348*pi) q[23];
U1q(3.4855635533359752*pi,1.3259936574649782*pi) q[24];
U1q(0.448421379806656*pi,1.461693473226258*pi) q[25];
U1q(0.447214703073184*pi,1.380408868685755*pi) q[26];
U1q(3.6270876555447*pi,0.49337808696790936*pi) q[27];
U1q(0.633479142932198*pi,0.944793694591626*pi) q[28];
U1q(0.664377000400492*pi,1.894765899865152*pi) q[29];
U1q(3.705728304103554*pi,0.8881537971256017*pi) q[30];
U1q(0.184955555368719*pi,1.288617059284533*pi) q[31];
U1q(0.24444516920078*pi,1.483774841946258*pi) q[32];
U1q(1.27596130442061*pi,0.007102234978094327*pi) q[33];
U1q(3.445947749398941*pi,1.021913570594395*pi) q[34];
U1q(0.256068641271956*pi,1.690904914473422*pi) q[35];
U1q(1.41666117235895*pi,0.724950819451324*pi) q[36];
U1q(0.928639564993128*pi,1.324381951808868*pi) q[37];
U1q(0.649029046109528*pi,0.915444453602425*pi) q[38];
U1q(0.219677388663013*pi,0.574936517693144*pi) q[39];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[4],q[27];
RZZ(0.5*pi) q[5],q[7];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[12],q[29];
RZZ(0.5*pi) q[28],q[13];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[36],q[17];
RZZ(0.5*pi) q[30],q[18];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[32],q[35];
U1q(0.298423029873851*pi,1.0069851107818102*pi) q[0];
U1q(0.230183955110343*pi,0.3876010721727272*pi) q[1];
U1q(0.837454218155804*pi,1.2688202833580244*pi) q[2];
U1q(0.317951190945961*pi,1.91753941032026*pi) q[3];
U1q(0.176784203360359*pi,0.23226983685327562*pi) q[4];
U1q(0.073226476309617*pi,1.6343133317878*pi) q[5];
U1q(0.661366448127549*pi,0.026388124684381342*pi) q[6];
U1q(0.718964694294639*pi,1.63900631674447*pi) q[7];
U1q(0.761052405441308*pi,1.371852451631126*pi) q[8];
U1q(0.244666174283456*pi,0.7245298428956799*pi) q[9];
U1q(0.442974313154838*pi,0.7355784199766977*pi) q[10];
U1q(0.40907175652497*pi,0.9327161798414898*pi) q[11];
U1q(0.349991335146117*pi,1.88856477035403*pi) q[12];
U1q(0.527373268546209*pi,1.1706700531307193*pi) q[13];
U1q(0.552580338901363*pi,1.876150908659644*pi) q[14];
U1q(0.710048450380978*pi,0.8768191211884231*pi) q[15];
U1q(0.86672924889152*pi,0.2377240482837999*pi) q[16];
U1q(0.613254501612922*pi,0.5228361793275491*pi) q[17];
U1q(0.844390284184498*pi,1.9759676351067799*pi) q[18];
U1q(0.415478033408383*pi,0.4871936891255628*pi) q[19];
U1q(0.551150801935494*pi,1.695144760906822*pi) q[20];
U1q(0.879083910653287*pi,1.5636366372421326*pi) q[21];
U1q(0.235362084962258*pi,1.332486318057351*pi) q[22];
U1q(0.119016768398773*pi,0.6713445789275001*pi) q[23];
U1q(0.649687421714332*pi,1.2716250345079878*pi) q[24];
U1q(0.458168171232826*pi,0.4500541728585201*pi) q[25];
U1q(0.773099235050097*pi,1.4060840573764999*pi) q[26];
U1q(0.337129176379781*pi,0.06496504307381934*pi) q[27];
U1q(0.728983134261213*pi,0.14305572099784003*pi) q[28];
U1q(0.630002251459235*pi,0.5165249174389399*pi) q[29];
U1q(0.591170816365943*pi,0.7670851146724417*pi) q[30];
U1q(0.445016929587138*pi,1.17890497733607*pi) q[31];
U1q(0.441948635640565*pi,1.21272875448272*pi) q[32];
U1q(0.817118680116741*pi,1.9167138558568646*pi) q[33];
U1q(0.714241028407645*pi,1.130389535116505*pi) q[34];
U1q(0.389211108432876*pi,1.50603044249759*pi) q[35];
U1q(0.77744174534176*pi,0.8464549796665239*pi) q[36];
U1q(0.542683668164774*pi,1.85264672872339*pi) q[37];
U1q(0.153787056309779*pi,1.96340595798205*pi) q[38];
U1q(0.897864767716764*pi,0.02109545560814996*pi) q[39];
RZZ(0.5*pi) q[0],q[17];
RZZ(0.5*pi) q[18],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[23],q[6];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[8],q[32];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[10],q[13];
RZZ(0.5*pi) q[11],q[19];
RZZ(0.5*pi) q[12],q[35];
RZZ(0.5*pi) q[36],q[14];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[26],q[25];
RZZ(0.5*pi) q[28],q[30];
RZZ(0.5*pi) q[34],q[37];
U1q(0.811763936547256*pi,0.44287031685477984*pi) q[0];
U1q(0.730244442299986*pi,0.08706944748539724*pi) q[1];
U1q(0.799760097723242*pi,0.7749548684071046*pi) q[2];
U1q(0.755484198351887*pi,0.8458108693903501*pi) q[3];
U1q(0.717031383045011*pi,0.3521544247920554*pi) q[4];
U1q(0.193124207893424*pi,0.9580279579915496*pi) q[5];
U1q(0.574783096088609*pi,1.457643831853911*pi) q[6];
U1q(0.054540420574346*pi,0.23884226084151017*pi) q[7];
U1q(0.591068859763034*pi,0.39049292890812604*pi) q[8];
U1q(0.779036606133117*pi,1.9436825414215804*pi) q[9];
U1q(0.350614731562988*pi,0.08780928575306746*pi) q[10];
U1q(0.29148713811182*pi,0.4208892113791096*pi) q[11];
U1q(0.424833896658406*pi,0.17561360078845034*pi) q[12];
U1q(0.54298745961549*pi,1.766633510370169*pi) q[13];
U1q(0.441121184315471*pi,1.9417632484120135*pi) q[14];
U1q(0.544536837801216*pi,0.21630700969886352*pi) q[15];
U1q(0.871438574048754*pi,0.057733332251749925*pi) q[16];
U1q(0.557776163775185*pi,0.8554656846494391*pi) q[17];
U1q(0.758982629596412*pi,1.17802264039962*pi) q[18];
U1q(0.664016688795224*pi,1.3910309551312432*pi) q[19];
U1q(0.91535203184552*pi,0.6768016621670552*pi) q[20];
U1q(0.429796895171319*pi,0.9047544321326919*pi) q[21];
U1q(0.703443992809807*pi,0.22143884112363077*pi) q[22];
U1q(0.373607567294336*pi,1.5033008671863701*pi) q[23];
U1q(0.404314230389637*pi,0.8225021629975879*pi) q[24];
U1q(0.632914852150206*pi,0.3727003734895096*pi) q[25];
U1q(0.615836065539139*pi,1.5301493561426902*pi) q[26];
U1q(0.334636192827553*pi,0.8859067346866594*pi) q[27];
U1q(0.350882926383352*pi,1.2665468042783399*pi) q[28];
U1q(0.935347444209399*pi,1.4584578301763402*pi) q[29];
U1q(0.551827232797139*pi,0.8620474528262616*pi) q[30];
U1q(0.496288683867586*pi,1.2377509653364296*pi) q[31];
U1q(0.325220922727523*pi,1.9518433590535702*pi) q[32];
U1q(0.679928863859406*pi,0.5656434293196946*pi) q[33];
U1q(0.433125759191451*pi,0.7876395708511854*pi) q[34];
U1q(0.558967946739991*pi,1.80267569418685*pi) q[35];
U1q(0.786355909118189*pi,0.562559448130104*pi) q[36];
U1q(0.299491212642653*pi,0.6142113294418201*pi) q[37];
U1q(0.747638411946992*pi,1.72111974785091*pi) q[38];
U1q(0.861111578366521*pi,1.2382828163083301*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[4],q[7];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[38],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[17];
RZZ(0.5*pi) q[33],q[18];
RZZ(0.5*pi) q[28],q[20];
RZZ(0.5*pi) q[23],q[21];
RZZ(0.5*pi) q[39],q[27];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[34],q[32];
U1q(0.309653943214499*pi,0.7757224605205497*pi) q[0];
U1q(0.818501669953447*pi,1.4564624221264175*pi) q[1];
U1q(0.267671976148395*pi,1.1024975620671649*pi) q[2];
U1q(0.890953751089609*pi,1.1526435905942503*pi) q[3];
U1q(0.449891852614868*pi,1.6863048456938454*pi) q[4];
U1q(0.670959449544745*pi,0.040457059810590223*pi) q[5];
U1q(0.439671473684151*pi,0.026554045932760673*pi) q[6];
U1q(0.878854332399132*pi,0.5150377370807098*pi) q[7];
U1q(0.608880981639745*pi,1.9883411188339952*pi) q[8];
U1q(0.518948484666197*pi,0.9121951440854597*pi) q[9];
U1q(0.451846476624711*pi,0.3919532006968369*pi) q[10];
U1q(0.484302793174408*pi,1.5489458427670897*pi) q[11];
U1q(0.24546307517804*pi,0.45250123584249025*pi) q[12];
U1q(0.49022737853396*pi,0.40149101647382857*pi) q[13];
U1q(0.344489788339027*pi,0.36968825703530417*pi) q[14];
U1q(0.607672527663914*pi,0.7211974332561129*pi) q[15];
U1q(0.731993061193721*pi,0.3343172788227502*pi) q[16];
U1q(0.220209288405259*pi,0.5029974900575289*pi) q[17];
U1q(0.256662241145993*pi,0.5632544019557004*pi) q[18];
U1q(0.321441685062388*pi,1.3849417446026333*pi) q[19];
U1q(0.648629215847144*pi,1.0221648553691458*pi) q[20];
U1q(0.712773204378671*pi,1.1879721558864738*pi) q[21];
U1q(0.506683832459213*pi,1.5390602642318312*pi) q[22];
U1q(0.0823900966685561*pi,0.3718883138250897*pi) q[23];
U1q(0.475797568224647*pi,0.9049285317788787*pi) q[24];
U1q(0.655014115082114*pi,0.9120582367591696*pi) q[25];
U1q(0.311381784649868*pi,0.7183028923685697*pi) q[26];
U1q(0.156879204640009*pi,0.460934519703339*pi) q[27];
U1q(0.855493792652027*pi,0.13095183665317034*pi) q[28];
U1q(0.354446958826845*pi,0.5404823760955901*pi) q[29];
U1q(0.488025038065903*pi,1.3505919893092315*pi) q[30];
U1q(0.060087631390388*pi,0.7974917528121992*pi) q[31];
U1q(0.473495269907666*pi,1.5395640774841404*pi) q[32];
U1q(0.192014303174918*pi,1.960940303705205*pi) q[33];
U1q(0.694476117267488*pi,0.09555748791354546*pi) q[34];
U1q(0.104692279220303*pi,0.5404962959739903*pi) q[35];
U1q(0.452999047917098*pi,1.4385973031790442*pi) q[36];
U1q(0.614473188192915*pi,0.5763200847612904*pi) q[37];
U1q(0.33014027899204*pi,1.2772254744667997*pi) q[38];
U1q(0.808631248298468*pi,0.42584824314312986*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[28],q[7];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[33],q[12];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[36],q[19];
RZZ(0.5*pi) q[21],q[31];
RZZ(0.5*pi) q[22],q[29];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[38],q[30];
U1q(0.430769205342995*pi,1.1833967345090803*pi) q[0];
U1q(0.868565711086121*pi,0.5388809784024877*pi) q[1];
U1q(0.302970555994612*pi,1.0466775688256345*pi) q[2];
U1q(0.660222808729292*pi,1.02396071732705*pi) q[3];
U1q(0.249732573984437*pi,0.7523888999735657*pi) q[4];
U1q(0.265250266737546*pi,0.5619136190051996*pi) q[5];
U1q(0.283488632976698*pi,0.6589857572578595*pi) q[6];
U1q(0.695941641280631*pi,0.4176288451688208*pi) q[7];
U1q(0.519087312303308*pi,0.9051797446391756*pi) q[8];
U1q(0.552832939908445*pi,1.12979140750351*pi) q[9];
U1q(0.441943187479822*pi,1.6280234155165783*pi) q[10];
U1q(0.691589496711237*pi,1.4504569225771*pi) q[11];
U1q(0.127584966371161*pi,1.3949440779786997*pi) q[12];
U1q(0.32412265412013*pi,1.3308900194361986*pi) q[13];
U1q(0.263753130629801*pi,0.1916889870177343*pi) q[14];
U1q(0.371524340869315*pi,0.6472596918246918*pi) q[15];
U1q(0.396624079077505*pi,1.0467655187040101*pi) q[16];
U1q(0.527994789544592*pi,0.9843876521497492*pi) q[17];
U1q(0.565529571115549*pi,1.8723980285092008*pi) q[18];
U1q(0.747743493918812*pi,1.3965382414807817*pi) q[19];
U1q(0.88054666165596*pi,0.14601931495712606*pi) q[20];
U1q(0.939803649966671*pi,1.5449158731477937*pi) q[21];
U1q(0.229193125586816*pi,1.729649261398821*pi) q[22];
U1q(0.468946180544885*pi,0.22043163647487063*pi) q[23];
U1q(0.771541439740492*pi,0.5234810040486995*pi) q[24];
U1q(0.810405624069192*pi,0.9330644956968008*pi) q[25];
U1q(0.624699603623525*pi,0.40964640455990065*pi) q[26];
U1q(0.270933852523146*pi,1.5852043757900809*pi) q[27];
U1q(0.491443758748342*pi,1.6502111245098998*pi) q[28];
U1q(0.343217333824544*pi,0.007189090335380399*pi) q[29];
U1q(0.629777584829657*pi,0.5445200709576525*pi) q[30];
U1q(0.727310727652513*pi,0.9377599687739*pi) q[31];
U1q(0.580539318579698*pi,0.9518925146505008*pi) q[32];
U1q(0.241448163395651*pi,0.39027308989600407*pi) q[33];
U1q(0.594542002330236*pi,0.3318329195767138*pi) q[34];
U1q(0.206653926647096*pi,1.9133556481928995*pi) q[35];
U1q(0.652433147689873*pi,0.6620098377964645*pi) q[36];
U1q(0.460075033929983*pi,0.6194086367152796*pi) q[37];
U1q(0.485247850836057*pi,1.3934039442540005*pi) q[38];
U1q(0.805550518091229*pi,1.9463402092596*pi) q[39];
rz(2.9448663737861596*pi) q[0];
rz(1.584141474983742*pi) q[1];
rz(2.386456352033557*pi) q[2];
rz(3.1399488413944603*pi) q[3];
rz(0.3290117571010338*pi) q[4];
rz(0.40454082811757*pi) q[5];
rz(0.21438678992663895*pi) q[6];
rz(2.8066868411663997*pi) q[7];
rz(0.4137596914076145*pi) q[8];
rz(2.0364327998453398*pi) q[9];
rz(2.3694105019098224*pi) q[10];
rz(3.3709784072968*pi) q[11];
rz(2.127592658663101*pi) q[12];
rz(0.6698449366139698*pi) q[13];
rz(3.152232971533886*pi) q[14];
rz(2.845074242269508*pi) q[15];
rz(0.46784084734619924*pi) q[16];
rz(3.3641041033737*pi) q[17];
rz(2.0550270865429994*pi) q[18];
rz(3.987080396181377*pi) q[19];
rz(3.108904530717834*pi) q[20];
rz(1.0489216855404067*pi) q[21];
rz(0.25990005105206926*pi) q[22];
rz(2.5163164940474*pi) q[23];
rz(1.1389520992007807*pi) q[24];
rz(2.0514189891836008*pi) q[25];
rz(1.4964497678510895*pi) q[26];
rz(0.3182915225477103*pi) q[27];
rz(2.351831193810799*pi) q[28];
rz(0.3885017047086201*pi) q[29];
rz(1.006898737171758*pi) q[30];
rz(3.467731679041*pi) q[31];
rz(3.0434967370366994*pi) q[32];
rz(2.967024862972245*pi) q[33];
rz(1.5313127841282856*pi) q[34];
rz(3.3940457731861997*pi) q[35];
rz(1.2261071186614458*pi) q[36];
rz(3.63439033177585*pi) q[37];
rz(3.2961047848418*pi) q[38];
rz(3.58709001154224*pi) q[39];
U1q(0.430769205342995*pi,1.1282631082952461*pi) q[0];
U1q(1.86856571108612*pi,1.12302245338623*pi) q[1];
U1q(1.30297055599461*pi,0.433133920859191*pi) q[2];
U1q(3.660222808729292*pi,1.16390955872151*pi) q[3];
U1q(1.24973257398444*pi,0.0814006570746051*pi) q[4];
U1q(1.26525026673755*pi,1.9664544471227732*pi) q[5];
U1q(1.2834886329767*pi,1.873372547184581*pi) q[6];
U1q(1.69594164128063*pi,0.22431568633525*pi) q[7];
U1q(1.51908731230331*pi,0.31893943604679*pi) q[8];
U1q(0.552832939908445*pi,0.166224207348848*pi) q[9];
U1q(0.441943187479822*pi,0.99743391742637*pi) q[10];
U1q(1.69158949671124*pi,1.821435329873838*pi) q[11];
U1q(0.127584966371161*pi,0.5225367366417599*pi) q[12];
U1q(0.32412265412013*pi,1.000734956050179*pi) q[13];
U1q(0.263753130629801*pi,0.343921958551643*pi) q[14];
U1q(1.37152434086932*pi,0.4923339340941899*pi) q[15];
U1q(0.396624079077505*pi,0.514606366050212*pi) q[16];
U1q(0.527994789544592*pi,1.3484917555234461*pi) q[17];
U1q(0.565529571115549*pi,0.927425115052162*pi) q[18];
U1q(0.747743493918812*pi,0.383618637662158*pi) q[19];
U1q(0.88054666165596*pi,0.254923845674961*pi) q[20];
U1q(1.93980364996667*pi,1.593837558688177*pi) q[21];
U1q(3.229193125586816*pi,0.9895493124508901*pi) q[22];
U1q(0.468946180544885*pi,1.736748130522251*pi) q[23];
U1q(1.77154143974049*pi,0.662433103249479*pi) q[24];
U1q(0.810405624069192*pi,1.9844834848804127*pi) q[25];
U1q(0.624699603623525*pi,0.9060961724109899*pi) q[26];
U1q(3.2709338525231457*pi,0.903495898337788*pi) q[27];
U1q(0.491443758748342*pi,1.002042318320657*pi) q[28];
U1q(0.343217333824544*pi,1.395690795044001*pi) q[29];
U1q(1.62977758482966*pi,0.551418808129408*pi) q[30];
U1q(1.72731072765251*pi,1.405491647814911*pi) q[31];
U1q(0.580539318579698*pi,0.99538925168721*pi) q[32];
U1q(0.241448163395651*pi,0.357297952868254*pi) q[33];
U1q(1.59454200233024*pi,0.863145703705004*pi) q[34];
U1q(1.2066539266471*pi,0.30740142137913*pi) q[35];
U1q(0.652433147689873*pi,0.888116956457912*pi) q[36];
U1q(1.46007503392998*pi,1.253798968491132*pi) q[37];
U1q(0.485247850836057*pi,1.689508729095788*pi) q[38];
U1q(0.805550518091229*pi,0.533430220801849*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[28],q[7];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[33],q[12];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[36],q[19];
RZZ(0.5*pi) q[21],q[31];
RZZ(0.5*pi) q[22],q[29];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[38],q[30];
U1q(0.309653943214499*pi,0.72058883430671*pi) q[0];
U1q(3.181498330046553*pi,0.20544100966229567*pi) q[1];
U1q(3.732328023851605*pi,0.3773139276176653*pi) q[2];
U1q(1.89095375108961*pi,1.0352266854543117*pi) q[3];
U1q(3.550108147385132*pi,0.1474847113543225*pi) q[4];
U1q(3.670959449544746*pi,0.48791100631738704*pi) q[5];
U1q(3.560328526315849*pi,0.5058042585097172*pi) q[6];
U1q(3.878854332399133*pi,0.12690679442336314*pi) q[7];
U1q(3.391119018360255*pi,1.2357780618519714*pi) q[8];
U1q(1.5189484846662*pi,0.948627943930799*pi) q[9];
U1q(1.45184647662471*pi,0.76136370260666*pi) q[10];
U1q(1.48430279317441*pi,0.7229464096838047*pi) q[11];
U1q(0.24546307517804*pi,0.58009389450559*pi) q[12];
U1q(0.49022737853396*pi,1.0713359530878002*pi) q[13];
U1q(0.344489788339027*pi,0.5219212285692101*pi) q[14];
U1q(3.392327472336086*pi,1.4183961926627706*pi) q[15];
U1q(1.73199306119372*pi,1.802158126168953*pi) q[16];
U1q(1.22020928840526*pi,1.86710159343122*pi) q[17];
U1q(0.256662241145993*pi,0.61828148849869*pi) q[18];
U1q(0.321441685062388*pi,1.372022140784009*pi) q[19];
U1q(0.648629215847144*pi,0.13106938608698004*pi) q[20];
U1q(3.287226795621328*pi,1.9507812759495224*pi) q[21];
U1q(1.50668383245921*pi,1.1801383096178824*pi) q[22];
U1q(1.08239009666856*pi,0.88820480787247*pi) q[23];
U1q(3.524202431775353*pi,0.28098557551930525*pi) q[24];
U1q(0.655014115082114*pi,1.9634772259427837*pi) q[25];
U1q(1.31138178464987*pi,1.214752660219656*pi) q[26];
U1q(1.15687920464001*pi,1.0277657544245309*pi) q[27];
U1q(1.85549379265203*pi,1.48278303046393*pi) q[28];
U1q(0.354446958826845*pi,1.928984080804209*pi) q[29];
U1q(1.4880250380659*pi,1.7453468897778277*pi) q[30];
U1q(3.060087631390388*pi,0.5457598637766437*pi) q[31];
U1q(1.47349526990767*pi,0.58306081452081*pi) q[32];
U1q(0.192014303174918*pi,0.92796516667745*pi) q[33];
U1q(3.305523882732511*pi,0.0994211353681757*pi) q[34];
U1q(3.895307720779697*pi,0.6802607735980838*pi) q[35];
U1q(0.452999047917098*pi,0.6647044218404901*pi) q[36];
U1q(3.385526811807085*pi,1.2968875204451296*pi) q[37];
U1q(0.33014027899204*pi,1.573330259308609*pi) q[38];
U1q(1.80863124829847*pi,1.012938254685377*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[4],q[7];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[38],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[17];
RZZ(0.5*pi) q[33],q[18];
RZZ(0.5*pi) q[28],q[20];
RZZ(0.5*pi) q[23],q[21];
RZZ(0.5*pi) q[39],q[27];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[34],q[32];
U1q(0.811763936547256*pi,1.387736690640938*pi) q[0];
U1q(1.73024444229999*pi,0.5748339843033082*pi) q[1];
U1q(3.200239902276757*pi,1.7048566212777243*pi) q[2];
U1q(1.75548419835189*pi,1.728393964250412*pi) q[3];
U1q(1.71703138304501*pi,0.48163513225611365*pi) q[4];
U1q(0.193124207893424*pi,0.40548190449834787*pi) q[5];
U1q(3.57478309608861*pi,1.0747144725885658*pi) q[6];
U1q(0.054540420574346*pi,1.8507113181841732*pi) q[7];
U1q(1.59106885976303*pi,1.833626251777843*pi) q[8];
U1q(1.77903660613312*pi,1.9171405465946727*pi) q[9];
U1q(1.35061473156299*pi,1.0655076175504237*pi) q[10];
U1q(1.29148713811182*pi,1.5948897782958267*pi) q[11];
U1q(3.4248338966584058*pi,1.3032062594515499*pi) q[12];
U1q(0.54298745961549*pi,1.4364784469841396*pi) q[13];
U1q(1.44112118431547*pi,0.09399621994591989*pi) q[14];
U1q(1.54453683780122*pi,0.9232866162200168*pi) q[15];
U1q(1.87143857404875*pi,1.078742072739952*pi) q[16];
U1q(3.442223836224815*pi,1.5146333988393108*pi) q[17];
U1q(0.758982629596412*pi,0.23304972694260995*pi) q[18];
U1q(0.664016688795224*pi,1.3781113513126169*pi) q[19];
U1q(3.9153520318455204*pi,1.78570619288489*pi) q[20];
U1q(3.570203104828681*pi,1.2339989997033025*pi) q[21];
U1q(1.70344399280981*pi,1.8625168865096873*pi) q[22];
U1q(3.6263924327056642*pi,1.7567922545111916*pi) q[23];
U1q(3.595685769610363*pi,1.3634119443005894*pi) q[24];
U1q(0.632914852150206*pi,0.4241193626731301*pi) q[25];
U1q(3.38416393446086*pi,1.4029061964455387*pi) q[26];
U1q(0.334636192827553*pi,1.4527379694078533*pi) q[27];
U1q(1.35088292638335*pi,0.34718806283876225*pi) q[28];
U1q(1.9353474442094*pi,0.8469595348849599*pi) q[29];
U1q(0.551827232797139*pi,1.2568023532948598*pi) q[30];
U1q(0.496288683867586*pi,1.9860190763008738*pi) q[31];
U1q(1.32522092272752*pi,1.1707815329513727*pi) q[32];
U1q(0.679928863859406*pi,0.5326682922919499*pi) q[33];
U1q(1.43312575919145*pi,1.4073390524305391*pi) q[34];
U1q(3.441032053260008*pi,1.4180813753852248*pi) q[35];
U1q(1.78635590911819*pi,0.78866656679156*pi) q[36];
U1q(3.700508787357347*pi,0.2589962757645896*pi) q[37];
U1q(0.747638411946992*pi,0.0172245326927187*pi) q[38];
U1q(1.86111157836652*pi,0.2005036815201744*pi) q[39];
RZZ(0.5*pi) q[0],q[17];
RZZ(0.5*pi) q[18],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[23],q[6];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[8],q[32];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[10],q[13];
RZZ(0.5*pi) q[11],q[19];
RZZ(0.5*pi) q[12],q[35];
RZZ(0.5*pi) q[36],q[14];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[26],q[25];
RZZ(0.5*pi) q[28],q[30];
RZZ(0.5*pi) q[34],q[37];
U1q(0.298423029873851*pi,1.9518514845679702*pi) q[0];
U1q(3.230183955110343*pi,1.875365608990648*pi) q[1];
U1q(1.8374542181558*pi,0.21099120632680768*pi) q[2];
U1q(3.682048809054038*pi,0.6566654233204998*pi) q[3];
U1q(0.176784203360359*pi,0.36175054431733367*pi) q[4];
U1q(1.07322647630962*pi,1.081767278294604*pi) q[5];
U1q(0.661366448127549*pi,1.6434587654190347*pi) q[6];
U1q(0.718964694294639*pi,0.250875374087133*pi) q[7];
U1q(1.76105240544131*pi,1.814985774500851*pi) q[8];
U1q(1.24466617428346*pi,1.697987848068764*pi) q[9];
U1q(0.442974313154838*pi,0.7132767517740537*pi) q[10];
U1q(3.59092824347503*pi,0.08306280983345538*pi) q[11];
U1q(3.6500086648538828*pi,1.5902550898859715*pi) q[12];
U1q(3.527373268546209*pi,1.8405149897447002*pi) q[13];
U1q(3.447419661098637*pi,0.15960855969828724*pi) q[14];
U1q(1.71004845038098*pi,0.5837987277095789*pi) q[15];
U1q(0.86672924889152*pi,0.25873278877201167*pi) q[16];
U1q(3.3867454983870777*pi,1.8472629041612008*pi) q[17];
U1q(1.8443902841845*pi,0.030994721649769907*pi) q[18];
U1q(0.415478033408383*pi,0.47427408530694004*pi) q[19];
U1q(3.448849198064506*pi,1.767363094145122*pi) q[20];
U1q(1.87908391065329*pi,0.5751167945938669*pi) q[21];
U1q(3.235362084962258*pi,1.7514694095759626*pi) q[22];
U1q(1.11901676839877*pi,1.5887485427700572*pi) q[23];
U1q(3.350312578285668*pi,0.9142890727901891*pi) q[24];
U1q(0.458168171232826*pi,0.50147316204214*pi) q[25];
U1q(3.773099235050096*pi,0.5269714952117246*pi) q[26];
U1q(0.337129176379781*pi,1.6317962777950132*pi) q[27];
U1q(1.72898313426121*pi,0.2236969795582624*pi) q[28];
U1q(3.369997748540765*pi,1.7888924476223602*pi) q[29];
U1q(0.591170816365943*pi,0.16184001514103974*pi) q[30];
U1q(0.445016929587138*pi,0.9271730883005138*pi) q[31];
U1q(3.441948635640565*pi,1.4316669283805128*pi) q[32];
U1q(1.81711868011674*pi,0.8837387188291101*pi) q[33];
U1q(1.71424102840765*pi,1.7500890166958492*pi) q[34];
U1q(3.389211108432876*pi,1.714726627074484*pi) q[35];
U1q(1.77744174534176*pi,1.5047710352551462*pi) q[36];
U1q(1.54268366816477*pi,0.020560876483029178*pi) q[37];
U1q(0.153787056309779*pi,1.2595107428238599*pi) q[38];
U1q(3.897864767716764*pi,0.9833163208199862*pi) q[39];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[4],q[27];
RZZ(0.5*pi) q[5],q[7];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[12],q[29];
RZZ(0.5*pi) q[28],q[13];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[36],q[17];
RZZ(0.5*pi) q[30],q[18];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[32],q[35];
U1q(0.886525246861671*pi,1.8402363258713001*pi) q[0];
U1q(3.552215943833609*pi,1.2248747595405622*pi) q[1];
U1q(0.629161628496821*pi,0.2119419288336939*pi) q[2];
U1q(1.43776540713171*pi,0.7036644699377792*pi) q[3];
U1q(0.698213887630589*pi,1.5535135014725237*pi) q[4];
U1q(1.11654584764675*pi,1.4106993741836085*pi) q[5];
U1q(0.911476377584166*pi,1.625579007903415*pi) q[6];
U1q(0.907465179521111*pi,0.7043563541499633*pi) q[7];
U1q(1.3868246501595*pi,0.18013258048283554*pi) q[8];
U1q(1.26263759106315*pi,1.7644617744587534*pi) q[9];
U1q(0.70683973765579*pi,1.7924299863247342*pi) q[10];
U1q(1.42741552571096*pi,1.0801416658728082*pi) q[11];
U1q(1.15262505889215*pi,1.2500396588495137*pi) q[12];
U1q(1.81219473895942*pi,0.9895897948214918*pi) q[13];
U1q(1.70113998675811*pi,1.7906026693495214*pi) q[14];
U1q(1.2885933677392*pi,1.4432221455189143*pi) q[15];
U1q(0.337772524803366*pi,1.0487173334999418*pi) q[16];
U1q(1.43085337843004*pi,0.7472235028627265*pi) q[17];
U1q(1.12473518225158*pi,1.6301113471926012*pi) q[18];
U1q(0.751145597841856*pi,1.9255858589552597*pi) q[19];
U1q(1.62239199193577*pi,1.1837403231610102*pi) q[20];
U1q(0.125928044502575*pi,0.4594938854051269*pi) q[21];
U1q(0.345413021103238*pi,0.032846579917512386*pi) q[22];
U1q(0.483652530379463*pi,0.9003271821788976*pi) q[23];
U1q(3.4855635533359752*pi,0.8599204498332023*pi) q[24];
U1q(0.448421379806656*pi,1.51311246240987*pi) q[25];
U1q(0.447214703073184*pi,0.5012963065209848*pi) q[26];
U1q(0.6270876555447*pi,1.0602093216891033*pi) q[27];
U1q(3.633479142932198*pi,0.4219590059644762*pi) q[28];
U1q(1.66437700040049*pi,0.4106514651961439*pi) q[29];
U1q(0.705728304103554*pi,0.2829086975941997*pi) q[30];
U1q(0.184955555368719*pi,1.0368851702489836*pi) q[31];
U1q(1.24444516920078*pi,1.1606208409169714*pi) q[32];
U1q(1.27596130442061*pi,1.7933503397078816*pi) q[33];
U1q(1.44594774939894*pi,0.8585649812179597*pi) q[34];
U1q(0.256068641271956*pi,1.899601099050317*pi) q[35];
U1q(0.416661172358953*pi,1.383266875039947*pi) q[36];
U1q(0.928639564993128*pi,0.49229609956849885*pi) q[37];
U1q(0.649029046109528*pi,0.2115492384442299*pi) q[38];
U1q(1.21967738866301*pi,0.429475258734989*pi) q[39];
rz(2.1597636741287*pi) q[0];
rz(2.7751252404594378*pi) q[1];
rz(1.788058071166306*pi) q[2];
rz(3.296335530062221*pi) q[3];
rz(2.4464864985274763*pi) q[4];
rz(0.5893006258163915*pi) q[5];
rz(2.374420992096585*pi) q[6];
rz(3.2956436458500367*pi) q[7];
rz(1.8198674195171645*pi) q[8];
rz(2.2355382255412466*pi) q[9];
rz(0.20757001367526584*pi) q[10];
rz(0.9198583341271918*pi) q[11];
rz(0.7499603411504863*pi) q[12];
rz(3.010410205178508*pi) q[13];
rz(0.2093973306504786*pi) q[14];
rz(0.5567778544810857*pi) q[15];
rz(2.951282666500058*pi) q[16];
rz(1.2527764971372735*pi) q[17];
rz(0.3698886528073988*pi) q[18];
rz(0.07441414104474031*pi) q[19];
rz(2.81625967683899*pi) q[20];
rz(1.540506114594873*pi) q[21];
rz(3.9671534200824876*pi) q[22];
rz(3.0996728178211024*pi) q[23];
rz(1.1400795501667977*pi) q[24];
rz(2.48688753759013*pi) q[25];
rz(1.4987036934790152*pi) q[26];
rz(2.9397906783108967*pi) q[27];
rz(3.578040994035524*pi) q[28];
rz(1.589348534803856*pi) q[29];
rz(1.7170913024058003*pi) q[30];
rz(2.9631148297510164*pi) q[31];
rz(2.8393791590830286*pi) q[32];
rz(0.2066496602921184*pi) q[33];
rz(3.1414350187820403*pi) q[34];
rz(2.100398900949683*pi) q[35];
rz(0.616733124960053*pi) q[36];
rz(3.507703900431501*pi) q[37];
rz(3.78845076155577*pi) q[38];
rz(3.570524741265011*pi) q[39];
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
