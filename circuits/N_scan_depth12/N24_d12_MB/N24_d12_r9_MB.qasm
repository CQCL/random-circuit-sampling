OPENQASM 2.0;
include "hqslib1.inc";

qreg q[24];
creg c[24];
U1q(1.36942832998999*pi,0.910753160484993*pi) q[0];
U1q(1.32803359760326*pi,0.02588726419555829*pi) q[1];
U1q(0.518858861508411*pi,0.984413273547299*pi) q[2];
U1q(0.331929900119393*pi,1.9076608411572724*pi) q[3];
U1q(1.69982804674572*pi,0.34787043740173224*pi) q[4];
U1q(0.336396775568051*pi,1.638652710977599*pi) q[5];
U1q(0.828557931656536*pi,1.888161574733559*pi) q[6];
U1q(0.247708398130156*pi,0.331503721585226*pi) q[7];
U1q(1.56036836220125*pi,0.8788336187789779*pi) q[8];
U1q(1.37976681070498*pi,1.7275378632158014*pi) q[9];
U1q(1.64546816433668*pi,1.7209197268341176*pi) q[10];
U1q(0.547648761171127*pi,0.684956812597893*pi) q[11];
U1q(0.241057318151576*pi,1.209328932100334*pi) q[12];
U1q(0.807060699563353*pi,0.165477323958297*pi) q[13];
U1q(0.345712726593747*pi,0.3182450501668499*pi) q[14];
U1q(1.8283452421454*pi,0.3601602392616908*pi) q[15];
U1q(1.55012367335084*pi,0.47242290569898665*pi) q[16];
U1q(1.47290428870044*pi,1.7583820361501863*pi) q[17];
U1q(0.605685623189422*pi,0.112431625414354*pi) q[18];
U1q(1.56139088274196*pi,0.3898097369139865*pi) q[19];
U1q(0.981275909619538*pi,1.925469994544566*pi) q[20];
U1q(0.102747590485315*pi,0.8285942331896301*pi) q[21];
U1q(1.22822674461014*pi,1.761946807936959*pi) q[22];
U1q(1.21284599465495*pi,0.535734846826632*pi) q[23];
RZZ(0.5*pi) q[23],q[0];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[3],q[20];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[12],q[13];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[18],q[19];
U1q(0.649536817203895*pi,0.172718875936556*pi) q[0];
U1q(0.323518937698512*pi,1.4337872450311782*pi) q[1];
U1q(0.385798716337255*pi,1.193814985218498*pi) q[2];
U1q(0.120088860250324*pi,0.7549566216332799*pi) q[3];
U1q(0.499170147755248*pi,0.7462978340258122*pi) q[4];
U1q(0.753805161878793*pi,1.62957265617842*pi) q[5];
U1q(0.600648184097455*pi,1.1922086888391301*pi) q[6];
U1q(0.283797674048734*pi,0.6705117102288098*pi) q[7];
U1q(0.597378379608578*pi,0.8680740662447679*pi) q[8];
U1q(0.339804138743373*pi,1.1025241732681907*pi) q[9];
U1q(0.82281455020888*pi,0.6681654374448076*pi) q[10];
U1q(0.786449400005408*pi,0.55777533447284*pi) q[11];
U1q(0.925497666676768*pi,1.29381493767302*pi) q[12];
U1q(0.510031007955721*pi,0.5946309878974501*pi) q[13];
U1q(0.794439550971829*pi,1.9756624546688304*pi) q[14];
U1q(0.849567842446811*pi,1.969827131654811*pi) q[15];
U1q(0.386333401632766*pi,0.40729497885363664*pi) q[16];
U1q(0.225418016643069*pi,0.6219627790230664*pi) q[17];
U1q(0.713950760365154*pi,1.0643020457154369*pi) q[18];
U1q(0.400526464478226*pi,1.8468066681529463*pi) q[19];
U1q(0.607872584578193*pi,0.3428091766248702*pi) q[20];
U1q(0.802671705996931*pi,1.6901367117998598*pi) q[21];
U1q(0.0606067369446544*pi,0.38775867120676866*pi) q[22];
U1q(0.355711540705484*pi,0.20143459823575194*pi) q[23];
RZZ(0.5*pi) q[0],q[16];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[3],q[17];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[11];
RZZ(0.5*pi) q[21],q[9];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[22],q[19];
U1q(0.396467275553313*pi,1.477109261758173*pi) q[0];
U1q(0.257140767384846*pi,0.3209014387758984*pi) q[1];
U1q(0.52911784728001*pi,1.2489075297091299*pi) q[2];
U1q(0.467197969601478*pi,1.84078863013913*pi) q[3];
U1q(0.420113260222829*pi,1.2043601088957416*pi) q[4];
U1q(0.965961743690607*pi,0.37406513020097965*pi) q[5];
U1q(0.165815454845658*pi,1.2500229329937396*pi) q[6];
U1q(0.644122610898929*pi,1.08246619051675*pi) q[7];
U1q(0.653843904682145*pi,1.2558992195655478*pi) q[8];
U1q(0.358282168152993*pi,1.5481028040948708*pi) q[9];
U1q(0.849708334056707*pi,0.7205474812438375*pi) q[10];
U1q(0.20582190300783*pi,0.21163682855618005*pi) q[11];
U1q(0.604083879200105*pi,1.89561581631237*pi) q[12];
U1q(0.613949893261019*pi,0.5569156720797701*pi) q[13];
U1q(0.499009679870351*pi,0.3084581073564099*pi) q[14];
U1q(0.136973159292752*pi,0.6727639908883911*pi) q[15];
U1q(0.517071398944405*pi,0.5541867105054568*pi) q[16];
U1q(0.34916135242342*pi,0.8269753236796058*pi) q[17];
U1q(0.719379826093557*pi,0.32802112776747006*pi) q[18];
U1q(0.727280468205317*pi,1.7641406688414563*pi) q[19];
U1q(0.693938224183313*pi,1.8880652606773003*pi) q[20];
U1q(0.606967457642358*pi,1.3467776708643697*pi) q[21];
U1q(0.402137286928923*pi,1.9776500434895494*pi) q[22];
U1q(0.708023194230944*pi,0.35334522640456223*pi) q[23];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[3],q[7];
RZZ(0.5*pi) q[12],q[4];
RZZ(0.5*pi) q[8],q[5];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[13],q[15];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[17],q[23];
U1q(0.742613825174502*pi,1.868252520783913*pi) q[0];
U1q(0.610234983561858*pi,0.18215002438965744*pi) q[1];
U1q(0.692012538843423*pi,1.1937422731716802*pi) q[2];
U1q(0.150087728701159*pi,0.9955684789631096*pi) q[3];
U1q(0.160244525345023*pi,0.3431433088103333*pi) q[4];
U1q(0.543658899855416*pi,1.8157641496191008*pi) q[5];
U1q(0.84344404017101*pi,1.4014924702182192*pi) q[6];
U1q(0.21999539187985*pi,1.3489877654730993*pi) q[7];
U1q(0.431277195272137*pi,0.8317883069155672*pi) q[8];
U1q(0.54429082917059*pi,1.783932028573032*pi) q[9];
U1q(0.651132867692958*pi,0.19215278307300743*pi) q[10];
U1q(0.140411607994639*pi,1.2335848796460702*pi) q[11];
U1q(0.738540287799181*pi,1.6816167606949604*pi) q[12];
U1q(0.668117160831216*pi,0.6115175570573497*pi) q[13];
U1q(0.295456126757532*pi,0.16743365334236948*pi) q[14];
U1q(0.527923013678305*pi,0.9194802595830414*pi) q[15];
U1q(0.562411667789389*pi,0.06040328547489704*pi) q[16];
U1q(0.780474626160709*pi,0.46060562805063565*pi) q[17];
U1q(0.334280769797394*pi,0.13406344184534014*pi) q[18];
U1q(0.520510227942373*pi,0.7859384225855761*pi) q[19];
U1q(0.639784605250734*pi,0.23983242940775007*pi) q[20];
U1q(0.114867439983355*pi,0.29929695293929015*pi) q[21];
U1q(0.542112327945451*pi,0.2602924000015294*pi) q[22];
U1q(0.4046413076256*pi,1.5272239157962115*pi) q[23];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[21],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[14];
RZZ(0.5*pi) q[20],q[5];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[12],q[15];
RZZ(0.5*pi) q[16],q[19];
U1q(0.27704960206993*pi,1.6957459123923115*pi) q[0];
U1q(0.593220762997249*pi,1.7405297910186786*pi) q[1];
U1q(0.524127209813112*pi,1.2512001999797793*pi) q[2];
U1q(0.370017152367312*pi,0.3687153656204103*pi) q[3];
U1q(0.826982335192615*pi,1.088947063893233*pi) q[4];
U1q(0.833014151288627*pi,1.6675687552254992*pi) q[5];
U1q(0.503423088789837*pi,0.4911598725739008*pi) q[6];
U1q(0.474903835949082*pi,1.4728880407415996*pi) q[7];
U1q(0.339700388046213*pi,0.10737670509156683*pi) q[8];
U1q(0.731593160866816*pi,0.7514740090768903*pi) q[9];
U1q(0.725995180767883*pi,0.33041771036747747*pi) q[10];
U1q(0.470984794822062*pi,1.8306871042976596*pi) q[11];
U1q(0.20729255667072*pi,0.21266161405763917*pi) q[12];
U1q(0.770035138620942*pi,1.4203221285223897*pi) q[13];
U1q(0.696704004799675*pi,1.7332545146499*pi) q[14];
U1q(0.671097312256013*pi,0.5495490781204406*pi) q[15];
U1q(0.26207048464851*pi,0.48886370098648513*pi) q[16];
U1q(0.754065390194999*pi,1.9242122379188356*pi) q[17];
U1q(0.323385618416491*pi,1.7082367669771799*pi) q[18];
U1q(0.750255072619027*pi,0.18077673145532547*pi) q[19];
U1q(0.0534759929420455*pi,0.5916270969951007*pi) q[20];
U1q(0.310637825851744*pi,0.4145663777176001*pi) q[21];
U1q(0.445974445502583*pi,0.12987392620393923*pi) q[22];
U1q(0.0671330892997706*pi,1.2411799557770413*pi) q[23];
RZZ(0.5*pi) q[0],q[20];
RZZ(0.5*pi) q[12],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[4],q[11];
RZZ(0.5*pi) q[9],q[5];
RZZ(0.5*pi) q[6],q[16];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[8],q[15];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[18],q[21];
U1q(0.509423524204858*pi,0.012172687202994581*pi) q[0];
U1q(0.683132335899822*pi,1.8040088770235592*pi) q[1];
U1q(0.583871716548418*pi,1.0769553926030007*pi) q[2];
U1q(0.717769851824331*pi,0.25498365735182027*pi) q[3];
U1q(0.526034622334185*pi,0.46889536398863285*pi) q[4];
U1q(0.524460730239844*pi,1.2992700721059993*pi) q[5];
U1q(0.426314922374548*pi,1.0530904554927005*pi) q[6];
U1q(0.224656147174357*pi,1.7903294332699993*pi) q[7];
U1q(0.207406805787389*pi,1.7762051174793783*pi) q[8];
U1q(0.586328366494071*pi,1.6307932990407004*pi) q[9];
U1q(0.635969315905881*pi,0.616583145952017*pi) q[10];
U1q(0.27285560504219*pi,0.21698310203156979*pi) q[11];
U1q(0.275448391266467*pi,0.28039948376440016*pi) q[12];
U1q(0.727757964078426*pi,0.24389486305175012*pi) q[13];
U1q(0.630388817257691*pi,1.5655023675637008*pi) q[14];
U1q(0.569408492334222*pi,1.7992236790501899*pi) q[15];
U1q(0.755014123514591*pi,1.8579061736539852*pi) q[16];
U1q(0.71211022861535*pi,1.2373773593945856*pi) q[17];
U1q(0.363679070650411*pi,1.4540115414023393*pi) q[18];
U1q(0.496728906586296*pi,0.19471736901528658*pi) q[19];
U1q(0.574167185006416*pi,0.1140625002867992*pi) q[20];
U1q(0.889948663528235*pi,1.8721298276657006*pi) q[21];
U1q(0.501588362158458*pi,0.16336051196850754*pi) q[22];
U1q(0.271902519643227*pi,0.1799191204265913*pi) q[23];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[1],q[19];
RZZ(0.5*pi) q[13],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[14],q[5];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[11],q[16];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[17],q[21];
U1q(0.44228613117399*pi,0.06466713071699459*pi) q[0];
U1q(0.466490710147268*pi,0.9965405891588581*pi) q[1];
U1q(0.317020972166848*pi,1.1016247045201997*pi) q[2];
U1q(0.123191475215878*pi,1.4941121786301998*pi) q[3];
U1q(0.544548989189521*pi,0.2039333261476326*pi) q[4];
U1q(0.482648650312716*pi,1.947315188473599*pi) q[5];
U1q(0.858188512850744*pi,1.9883245146014001*pi) q[6];
U1q(0.590424523028169*pi,0.24544953205669984*pi) q[7];
U1q(0.746605346149908*pi,1.4133262155116793*pi) q[8];
U1q(0.282474111222562*pi,1.6130465655618007*pi) q[9];
U1q(0.296400657659069*pi,1.4327295973717167*pi) q[10];
U1q(0.432069586475427*pi,1.0889641353094994*pi) q[11];
U1q(0.476533686459582*pi,0.5212199228115004*pi) q[12];
U1q(0.298812386783786*pi,0.2055259967167995*pi) q[13];
U1q(0.497112773584954*pi,0.029146677962300416*pi) q[14];
U1q(0.317898496742113*pi,0.54718925038439*pi) q[15];
U1q(0.55577101845764*pi,0.3012518080665867*pi) q[16];
U1q(0.438993938172907*pi,1.4682953728326869*pi) q[17];
U1q(0.779766120349911*pi,0.7708689695623807*pi) q[18];
U1q(0.27794138896965*pi,1.402350782403687*pi) q[19];
U1q(0.294414516595878*pi,0.3035184509503992*pi) q[20];
U1q(0.216786512755108*pi,1.4996820833809998*pi) q[21];
U1q(0.503613127534148*pi,0.6447951167719577*pi) q[22];
U1q(0.713728904378874*pi,1.6098457015446321*pi) q[23];
rz(3.5102949323576063*pi) q[0];
rz(3.9133942535256416*pi) q[1];
rz(3.6427575945143005*pi) q[2];
rz(0.3620772705848001*pi) q[3];
rz(2.7082879500308668*pi) q[4];
rz(3.389016971400901*pi) q[5];
rz(0.3391515147127002*pi) q[6];
rz(2.7424650846721015*pi) q[7];
rz(0.6677843779367212*pi) q[8];
rz(0.678352489808999*pi) q[9];
rz(2.3320266080706826*pi) q[10];
rz(1.5716406194101005*pi) q[11];
rz(0.7640909745708999*pi) q[12];
rz(3.644115618976601*pi) q[13];
rz(0.7992632490443015*pi) q[14];
rz(2.7426863269850106*pi) q[15];
rz(2.7719154494994136*pi) q[16];
rz(0.09851344938801354*pi) q[17];
rz(0.46751527180735053*pi) q[18];
rz(1.8735739704413135*pi) q[19];
rz(3.1015951072311*pi) q[20];
rz(3.7095639528351008*pi) q[21];
rz(2.1494226238198415*pi) q[22];
rz(2.498298333012368*pi) q[23];
U1q(1.44228613117399*pi,0.57496206307455*pi) q[0];
U1q(1.46649071014727*pi,1.909934842684529*pi) q[1];
U1q(0.317020972166848*pi,1.744382299034491*pi) q[2];
U1q(1.12319147521588*pi,0.85618944921497*pi) q[3];
U1q(1.54454898918952*pi,1.9122212761785224*pi) q[4];
U1q(1.48264865031272*pi,0.336332159874536*pi) q[5];
U1q(1.85818851285074*pi,1.3274760293141*pi) q[6];
U1q(0.590424523028169*pi,1.987914616728864*pi) q[7];
U1q(3.746605346149909*pi,1.08111059344842*pi) q[8];
U1q(1.28247411122256*pi,1.291399055370809*pi) q[9];
U1q(1.29640065765907*pi,0.7647562054424*pi) q[10];
U1q(0.432069586475427*pi,1.660604754719621*pi) q[11];
U1q(0.476533686459582*pi,0.285310897382356*pi) q[12];
U1q(0.298812386783786*pi,0.84964161569337*pi) q[13];
U1q(0.497112773584954*pi,1.8284099270065979*pi) q[14];
U1q(1.31789849674211*pi,0.289875577369425*pi) q[15];
U1q(0.55577101845764*pi,0.073167257565938*pi) q[16];
U1q(0.438993938172907*pi,0.566808822220672*pi) q[17];
U1q(1.77976612034991*pi,0.238384241369728*pi) q[18];
U1q(0.27794138896965*pi,0.275924752844914*pi) q[19];
U1q(1.29441451659588*pi,0.405113558181449*pi) q[20];
U1q(1.21678651275511*pi,0.209246036216057*pi) q[21];
U1q(1.50361312753415*pi,1.794217740591797*pi) q[22];
U1q(1.71372890437887*pi,1.10814403455698*pi) q[23];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[1],q[19];
RZZ(0.5*pi) q[13],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[14],q[5];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[11],q[16];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[17],q[21];
U1q(1.50942352420486*pi,1.6274565065885034*pi) q[0];
U1q(3.683132335899822*pi,1.1024665548198418*pi) q[1];
U1q(0.583871716548418*pi,0.71971298711731*pi) q[2];
U1q(1.71776985182433*pi,0.09531797049330049*pi) q[3];
U1q(3.473965377665815*pi,1.647259238337537*pi) q[4];
U1q(3.475539269760156*pi,1.9843772762421146*pi) q[5];
U1q(1.42631492237455*pi,1.262710088422824*pi) q[6];
U1q(0.224656147174357*pi,1.532794517942106*pi) q[7];
U1q(1.20740680578739*pi,0.7182316914807014*pi) q[8];
U1q(3.413671633505929*pi,1.2736523218919273*pi) q[9];
U1q(1.63596931590588*pi,0.5809026568620994*pi) q[10];
U1q(1.27285560504219*pi,1.7886237214416698*pi) q[11];
U1q(0.275448391266467*pi,0.0444904583353223*pi) q[12];
U1q(0.727757964078426*pi,1.888010482028365*pi) q[13];
U1q(1.63038881725769*pi,1.3647656166080302*pi) q[14];
U1q(3.430591507665778*pi,1.0378411487036514*pi) q[15];
U1q(0.755014123514591*pi,0.629821623153344*pi) q[16];
U1q(1.71211022861535*pi,0.3358908087825301*pi) q[17];
U1q(1.36367907065041*pi,0.555241669529778*pi) q[18];
U1q(1.4967289065863*pi,1.0682913394565992*pi) q[19];
U1q(1.57416718500642*pi,1.5945695088449952*pi) q[20];
U1q(1.88994866352824*pi,1.8367982919312944*pi) q[21];
U1q(3.498411637841542*pi,0.27565234539525374*pi) q[22];
U1q(1.27190251964323*pi,1.5380706156750166*pi) q[23];
RZZ(0.5*pi) q[0],q[20];
RZZ(0.5*pi) q[12],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[4],q[11];
RZZ(0.5*pi) q[9],q[5];
RZZ(0.5*pi) q[6],q[16];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[8],q[15];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[18],q[21];
U1q(1.27704960206993*pi,0.3110297317777724*pi) q[0];
U1q(1.59322076299725*pi,0.03898746881494253*pi) q[1];
U1q(0.524127209813112*pi,1.89395779449409*pi) q[2];
U1q(0.370017152367312*pi,1.2090496787618905*pi) q[3];
U1q(3.173017664807385*pi,0.02720753843287389*pi) q[4];
U1q(3.166985848711372*pi,0.6160785931227053*pi) q[5];
U1q(1.50342308878984*pi,1.7007795055040758*pi) q[6];
U1q(3.474903835949082*pi,1.21535312541373*pi) q[7];
U1q(0.339700388046213*pi,1.0494032790928662*pi) q[8];
U1q(3.268406839133184*pi,1.1529716118557212*pi) q[9];
U1q(1.72599518076788*pi,1.2947372212775186*pi) q[10];
U1q(1.47098479482206*pi,1.1749197191755796*pi) q[11];
U1q(0.20729255667072*pi,0.97675258862852*pi) q[12];
U1q(1.77003513862094*pi,1.064437747499*pi) q[13];
U1q(1.69670400479968*pi,0.1970134695218606*pi) q[14];
U1q(1.67109731225601*pi,1.2875157496333967*pi) q[15];
U1q(0.26207048464851*pi,0.260779150485906*pi) q[16];
U1q(3.2459346098050013*pi,0.6490559302582573*pi) q[17];
U1q(0.323385618416491*pi,1.8094668951046171*pi) q[18];
U1q(3.249744927380973*pi,0.08223197701661089*pi) q[19];
U1q(0.0534759929420455*pi,1.0721341055532563*pi) q[20];
U1q(1.31063782585174*pi,1.3792348419831535*pi) q[21];
U1q(1.44597444550258*pi,0.30913893115982183*pi) q[22];
U1q(0.0671330892997706*pi,1.5993314510254728*pi) q[23];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[21],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[14];
RZZ(0.5*pi) q[20],q[5];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[12],q[15];
RZZ(0.5*pi) q[16],q[19];
U1q(3.257386174825498*pi,0.13852312338617456*pi) q[0];
U1q(3.610234983561858*pi,1.597367235443961*pi) q[1];
U1q(0.692012538843423*pi,1.8364998676859896*pi) q[2];
U1q(0.150087728701159*pi,0.8359027921045907*pi) q[3];
U1q(3.839755474654976*pi,0.7730112935158531*pi) q[4];
U1q(3.456341100144584*pi,0.46788319872905526*pi) q[5];
U1q(1.84344404017101*pi,1.7904469078597591*pi) q[6];
U1q(3.78000460812015*pi,0.3392534006821739*pi) q[7];
U1q(0.431277195272137*pi,0.7738148809168612*pi) q[8];
U1q(3.45570917082941*pi,0.12051359235957929*pi) q[9];
U1q(3.348867132307042*pi,1.433002148571993*pi) q[10];
U1q(1.14041160799464*pi,0.5778174945239893*pi) q[11];
U1q(1.73854028779918*pi,1.4457077352658398*pi) q[12];
U1q(3.331882839168784*pi,1.8732423189640428*pi) q[13];
U1q(0.295456126757532*pi,0.6311926082143495*pi) q[14];
U1q(0.527923013678305*pi,0.6574469310960067*pi) q[15];
U1q(0.562411667789389*pi,1.8323187349742902*pi) q[16];
U1q(3.219525373839291*pi,0.11266254012646337*pi) q[17];
U1q(0.334280769797394*pi,0.23529356997278017*pi) q[18];
U1q(3.4794897720576268*pi,1.4770702858863611*pi) q[19];
U1q(1.63978460525073*pi,0.7203394379658963*pi) q[20];
U1q(3.114867439983355*pi,1.4945042667614605*pi) q[21];
U1q(0.542112327945451*pi,0.439557404957422*pi) q[22];
U1q(1.4046413076256*pi,1.8853754110446408*pi) q[23];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[3],q[7];
RZZ(0.5*pi) q[12],q[4];
RZZ(0.5*pi) q[8],q[5];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[13],q[15];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[17],q[23];
U1q(3.603532724446687*pi,0.5296663824119145*pi) q[0];
U1q(0.257140767384846*pi,0.736118649830191*pi) q[1];
U1q(1.52911784728001*pi,0.8916651242234499*pi) q[2];
U1q(1.46719796960148*pi,1.6811229432806094*pi) q[3];
U1q(1.42011326022283*pi,1.9117944934303934*pi) q[4];
U1q(3.034038256309392*pi,1.9095822181471769*pi) q[5];
U1q(0.165815454845658*pi,0.6389773706352821*pi) q[6];
U1q(3.355877389101072*pi,0.6057749756385737*pi) q[7];
U1q(1.65384390468214*pi,1.1979257935668413*pi) q[8];
U1q(1.35828216815299*pi,0.35634281683774294*pi) q[9];
U1q(3.150291665943293*pi,1.9046074504011736*pi) q[10];
U1q(1.20582190300783*pi,1.5997655456138844*pi) q[11];
U1q(3.604083879200105*pi,0.23170867964842445*pi) q[12];
U1q(3.386050106738981*pi,1.9278442039416257*pi) q[13];
U1q(0.499009679870351*pi,1.7722170622283908*pi) q[14];
U1q(0.136973159292752*pi,0.4107306624013467*pi) q[15];
U1q(0.517071398944405*pi,1.32610216000485*pi) q[16];
U1q(3.65083864757658*pi,0.7462928444974932*pi) q[17];
U1q(1.71937982609356*pi,1.429251255894915*pi) q[18];
U1q(3.272719531794683*pi,1.498868039630481*pi) q[19];
U1q(1.69393822418331*pi,0.07210660669634938*pi) q[20];
U1q(0.606967457642358*pi,0.5419849846865435*pi) q[21];
U1q(0.402137286928923*pi,0.15691504844544113*pi) q[22];
U1q(3.291976805769056*pi,1.0592541004363014*pi) q[23];
RZZ(0.5*pi) q[0],q[16];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[3],q[17];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[11];
RZZ(0.5*pi) q[21],q[9];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[22],q[19];
U1q(3.350463182796105*pi,1.8340567682335345*pi) q[0];
U1q(0.323518937698512*pi,1.8490044560854813*pi) q[1];
U1q(3.614201283662744*pi,1.946757668714083*pi) q[2];
U1q(3.879911139749676*pi,0.7669549517864613*pi) q[3];
U1q(0.499170147755248*pi,1.4537322185604733*pi) q[4];
U1q(1.75380516187879*pi,1.6540746921697382*pi) q[5];
U1q(0.600648184097455*pi,1.5811631264806723*pi) q[6];
U1q(1.28379767404873*pi,0.017729455926513538*pi) q[7];
U1q(3.402621620391421*pi,0.5857509468876252*pi) q[8];
U1q(1.33980413874337*pi,0.910764186011064*pi) q[9];
U1q(3.177185449791119*pi,0.9569894942002035*pi) q[10];
U1q(0.786449400005408*pi,1.9459040515305448*pi) q[11];
U1q(0.925497666676768*pi,1.6299078010090642*pi) q[12];
U1q(3.489968992044279*pi,1.8901288881239355*pi) q[13];
U1q(1.79443955097183*pi,1.4394214095408309*pi) q[14];
U1q(1.84956784244681*pi,1.7077938031677666*pi) q[15];
U1q(1.38633340163277*pi,1.1792104283530298*pi) q[16];
U1q(3.774581983356932*pi,1.951305389154033*pi) q[17];
U1q(3.286049239634845*pi,1.6929703379469503*pi) q[18];
U1q(1.40052646447823*pi,0.4162020403189848*pi) q[19];
U1q(0.607872584578193*pi,1.5268505226439189*pi) q[20];
U1q(1.80267170599693*pi,1.8853440256220355*pi) q[21];
U1q(1.06060673694465*pi,0.567023676162651*pi) q[22];
U1q(3.644288459294516*pi,0.2111647286051095*pi) q[23];
RZZ(0.5*pi) q[23],q[0];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[3],q[20];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[12],q[13];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[18],q[19];
U1q(1.36942832998999*pi,1.0960224836850894*pi) q[0];
U1q(0.328033597603263*pi,0.4411044752498503*pi) q[1];
U1q(1.51885886150841*pi,1.1561593803852812*pi) q[2];
U1q(1.33192990011939*pi,0.6142507322624766*pi) q[3];
U1q(0.699828046745717*pi,1.0553048219363834*pi) q[4];
U1q(0.336396775568051*pi,1.6631547469689223*pi) q[5];
U1q(0.828557931656536*pi,1.2771160123751022*pi) q[6];
U1q(0.247708398130156*pi,0.6787214672829336*pi) q[7];
U1q(1.56036836220125*pi,0.5749913943534124*pi) q[8];
U1q(1.37976681070498*pi,0.28575049606344893*pi) q[9];
U1q(1.64546816433668*pi,1.90423520481089*pi) q[10];
U1q(0.547648761171127*pi,0.07308552965560455*pi) q[11];
U1q(0.241057318151576*pi,1.5454217954363845*pi) q[12];
U1q(1.80706069956335*pi,1.319282552063097*pi) q[13];
U1q(3.3457127265937467*pi,1.0968388140428065*pi) q[14];
U1q(1.8283452421454*pi,0.3174606955608885*pi) q[15];
U1q(1.55012367335084*pi,0.11408250150767518*pi) q[16];
U1q(1.47290428870044*pi,1.814886132026905*pi) q[17];
U1q(1.60568562318942*pi,1.6448407582480322*pi) q[18];
U1q(0.561390882741962*pi,0.9592051090800249*pi) q[19];
U1q(0.981275909619538*pi,1.1095113405636194*pi) q[20];
U1q(3.102747590485315*pi,1.7468865042322717*pi) q[21];
U1q(1.22822674461014*pi,0.19283553943246012*pi) q[22];
U1q(1.21284599465495*pi,1.8768644800142282*pi) q[23];
rz(2.9039775163149106*pi) q[0];
rz(3.5588955247501497*pi) q[1];
rz(2.843840619614719*pi) q[2];
rz(3.3857492677375234*pi) q[3];
rz(0.9446951780636166*pi) q[4];
rz(2.3368452530310777*pi) q[5];
rz(0.7228839876248978*pi) q[6];
rz(3.3212785327170664*pi) q[7];
rz(3.4250086056465876*pi) q[8];
rz(1.714249503936551*pi) q[9];
rz(0.09576479518911007*pi) q[10];
rz(3.9269144703443954*pi) q[11];
rz(0.45457820456361553*pi) q[12];
rz(0.6807174479369029*pi) q[13];
rz(2.9031611859571935*pi) q[14];
rz(3.6825393044391115*pi) q[15];
rz(1.8859174984923248*pi) q[16];
rz(2.185113867973095*pi) q[17];
rz(0.35515924175196767*pi) q[18];
rz(1.0407948909199751*pi) q[19];
rz(0.8904886594363806*pi) q[20];
rz(0.25311349576772835*pi) q[21];
rz(3.80716446056754*pi) q[22];
rz(2.1231355199857718*pi) q[23];
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