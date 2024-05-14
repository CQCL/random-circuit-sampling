OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.40930381121135*pi,1.263555306383119*pi) q[0];
U1q(0.512259288312356*pi,1.2601191751198861*pi) q[1];
U1q(0.072667769773131*pi,1.683839617065879*pi) q[2];
U1q(1.69397670785088*pi,0.2691761947535232*pi) q[3];
U1q(0.217347225213229*pi,1.717648722482963*pi) q[4];
U1q(1.73364437128574*pi,1.3197447309090267*pi) q[5];
U1q(0.877657894072522*pi,0.535919201754281*pi) q[6];
U1q(1.8141422073031*pi,1.6696806041662333*pi) q[7];
U1q(0.875128873871043*pi,0.146202877736265*pi) q[8];
U1q(1.48317383890451*pi,0.3158703415344939*pi) q[9];
U1q(0.604782902755503*pi,1.9825598316488218*pi) q[10];
U1q(0.75194519417096*pi,0.376547735313324*pi) q[11];
U1q(0.628062213644457*pi,0.670247554037611*pi) q[12];
U1q(0.257468752306347*pi,1.072392900768375*pi) q[13];
U1q(1.53791457800741*pi,0.3310423066812678*pi) q[14];
U1q(0.024727804190621*pi,1.4992973711914162*pi) q[15];
U1q(0.658632000997557*pi,0.62420006355499*pi) q[16];
U1q(0.613385602594099*pi,0.761714786618275*pi) q[17];
U1q(1.2823072578989*pi,0.6345507005953188*pi) q[18];
U1q(0.198764731253661*pi,0.79627425068002*pi) q[19];
U1q(0.604644093263753*pi,1.38807444443167*pi) q[20];
U1q(0.92774843136428*pi,0.0908178131247394*pi) q[21];
U1q(0.169700944991006*pi,0.533116322483898*pi) q[22];
U1q(0.410841022524549*pi,0.115255859489352*pi) q[23];
U1q(0.364714896052908*pi,1.040543112534206*pi) q[24];
U1q(0.231689779809899*pi,1.45194669658066*pi) q[25];
U1q(1.61296809181655*pi,0.3090323068962104*pi) q[26];
U1q(1.48263541154819*pi,0.3153920422311072*pi) q[27];
U1q(0.599237093574775*pi,0.505179512322151*pi) q[28];
U1q(0.696510930820612*pi,0.589280131402669*pi) q[29];
U1q(0.613251084095671*pi,1.9258669537183377*pi) q[30];
U1q(1.42651259957532*pi,1.8761926175017594*pi) q[31];
U1q(3.405149512875348*pi,1.2622936669377753*pi) q[32];
U1q(1.73064435578518*pi,1.6687685626490953*pi) q[33];
U1q(0.550254184137239*pi,1.506761228897062*pi) q[34];
U1q(0.420188261033215*pi,0.443174653852482*pi) q[35];
U1q(0.182672337169625*pi,0.253273654040156*pi) q[36];
U1q(1.5398638841728*pi,0.21975871301571154*pi) q[37];
U1q(1.68868874220359*pi,1.3394768131279755*pi) q[38];
U1q(0.7639944310855*pi,1.785399595042933*pi) q[39];
RZZ(0.5*pi) q[29],q[0];
RZZ(0.5*pi) q[35],q[1];
RZZ(0.5*pi) q[26],q[2];
RZZ(0.5*pi) q[3],q[17];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[12],q[6];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[38];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[14],q[18];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[16],q[28];
RZZ(0.5*pi) q[19],q[21];
RZZ(0.5*pi) q[20],q[39];
RZZ(0.5*pi) q[22],q[24];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[25],q[37];
U1q(0.335911659108376*pi,0.7419180550586599*pi) q[0];
U1q(0.60271166981823*pi,1.2866197239538*pi) q[1];
U1q(0.183388075027245*pi,1.5251072691294798*pi) q[2];
U1q(0.974565594444938*pi,0.17734583865459297*pi) q[3];
U1q(0.166834012347452*pi,1.15900952616856*pi) q[4];
U1q(0.152630862914803*pi,0.19434060997322655*pi) q[5];
U1q(0.539200773277447*pi,0.4067264944673099*pi) q[6];
U1q(0.901898458521014*pi,0.4303864284543337*pi) q[7];
U1q(0.281902015779044*pi,0.9830774871847501*pi) q[8];
U1q(0.523763207869831*pi,1.226959824043024*pi) q[9];
U1q(0.377693307328995*pi,1.16086353127359*pi) q[10];
U1q(0.827051950085106*pi,1.269845027032721*pi) q[11];
U1q(0.566977859533319*pi,1.726275279475777*pi) q[12];
U1q(0.927815230700317*pi,0.3678604858104*pi) q[13];
U1q(0.341247866575468*pi,1.728488551007878*pi) q[14];
U1q(0.648320300030541*pi,0.5335976324345801*pi) q[15];
U1q(0.426899309430264*pi,1.8312762154584101*pi) q[16];
U1q(0.734661430104577*pi,1.367978543503056*pi) q[17];
U1q(0.15344294931327*pi,0.048041835708741765*pi) q[18];
U1q(0.528226487399621*pi,1.5618880595331701*pi) q[19];
U1q(0.117633769175637*pi,1.222170398845905*pi) q[20];
U1q(0.711441447251137*pi,1.5570856189762199*pi) q[21];
U1q(0.66278702211824*pi,1.9812753364864202*pi) q[22];
U1q(0.830079556254018*pi,0.1454559270354201*pi) q[23];
U1q(0.886463222403946*pi,0.9002425870364901*pi) q[24];
U1q(0.619119957674733*pi,0.6866675330302399*pi) q[25];
U1q(0.0594456235766427*pi,1.8451935342415204*pi) q[26];
U1q(0.70607011768905*pi,1.2948070865998473*pi) q[27];
U1q(0.70713865915301*pi,1.7247839433378989*pi) q[28];
U1q(0.730212891504842*pi,1.067179131784395*pi) q[29];
U1q(0.647872082243359*pi,0.20138178193215994*pi) q[30];
U1q(0.320269116937998*pi,1.5349517276811993*pi) q[31];
U1q(0.233741906855292*pi,0.4046794410787653*pi) q[32];
U1q(0.244457727509319*pi,0.6174084153546056*pi) q[33];
U1q(0.594846733391055*pi,0.27792127776216*pi) q[34];
U1q(0.398913015694122*pi,1.6665559125658298*pi) q[35];
U1q(0.325405955830916*pi,1.0281448314309798*pi) q[36];
U1q(0.479977969494632*pi,1.0800552888140014*pi) q[37];
U1q(0.559951067880337*pi,1.7958174914345055*pi) q[38];
U1q(0.652948675742599*pi,1.15031221054549*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[3],q[4];
RZZ(0.5*pi) q[20],q[5];
RZZ(0.5*pi) q[6],q[35];
RZZ(0.5*pi) q[22],q[7];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[10],q[32];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[12],q[34];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[26],q[15];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[38],q[23];
RZZ(0.5*pi) q[29],q[24];
RZZ(0.5*pi) q[25],q[36];
RZZ(0.5*pi) q[39],q[33];
U1q(0.879357969208414*pi,0.28996074952351014*pi) q[0];
U1q(0.953818036163419*pi,1.32164701614838*pi) q[1];
U1q(0.60053894815482*pi,0.7169025904769999*pi) q[2];
U1q(0.644946876409075*pi,1.5809454483887735*pi) q[3];
U1q(0.26750943881136*pi,0.8652538707308199*pi) q[4];
U1q(0.601224290219005*pi,0.7243523708693367*pi) q[5];
U1q(0.444893486897357*pi,1.3414077867609198*pi) q[6];
U1q(0.482328190782587*pi,0.2482865570843531*pi) q[7];
U1q(0.622123393958704*pi,0.89357970506647*pi) q[8];
U1q(0.623688828415841*pi,0.9416854224467848*pi) q[9];
U1q(0.677610837615689*pi,1.8735724477933404*pi) q[10];
U1q(0.804901502544758*pi,0.59697806888208*pi) q[11];
U1q(0.366055391870163*pi,0.5720734471750499*pi) q[12];
U1q(0.385997861910066*pi,1.5029039859884303*pi) q[13];
U1q(0.33578436202363*pi,1.1922783513391284*pi) q[14];
U1q(0.741899832561837*pi,0.57831818426006*pi) q[15];
U1q(0.707044844032274*pi,0.5977028487091003*pi) q[16];
U1q(0.641704417404915*pi,0.64459133409725*pi) q[17];
U1q(0.0884609881801889*pi,1.5423463886265782*pi) q[18];
U1q(0.331469517919423*pi,1.4502055037509196*pi) q[19];
U1q(0.513107240936898*pi,0.35668277234767*pi) q[20];
U1q(0.733520201793103*pi,0.01729287530510959*pi) q[21];
U1q(0.535506186557007*pi,0.3834051788903001*pi) q[22];
U1q(0.698688497412399*pi,1.6387058748839598*pi) q[23];
U1q(0.838688477986036*pi,0.8002102948066803*pi) q[24];
U1q(0.652162429422769*pi,1.2487413118507202*pi) q[25];
U1q(0.579549155373181*pi,1.9085163016466105*pi) q[26];
U1q(0.844902707040219*pi,0.8432596933902974*pi) q[27];
U1q(0.14575248303595*pi,0.5186282822940602*pi) q[28];
U1q(0.534114791604311*pi,0.7931881661859599*pi) q[29];
U1q(0.577423295132943*pi,1.3590655192753802*pi) q[30];
U1q(0.437968957664405*pi,0.9634691332340397*pi) q[31];
U1q(0.601531199488112*pi,1.953232838017696*pi) q[32];
U1q(0.627116011480857*pi,0.25717471652247514*pi) q[33];
U1q(0.493875894943937*pi,1.02632617700608*pi) q[34];
U1q(0.619141256015558*pi,1.4472212565238802*pi) q[35];
U1q(0.142961479990573*pi,1.5126579379211096*pi) q[36];
U1q(0.690462161211037*pi,0.7575842746829418*pi) q[37];
U1q(0.762218059196089*pi,0.5567906069665853*pi) q[38];
U1q(0.404326876821398*pi,1.4993197887130503*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[20],q[2];
RZZ(0.5*pi) q[4],q[37];
RZZ(0.5*pi) q[5],q[32];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[9],q[26];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[12],q[21];
RZZ(0.5*pi) q[17],q[13];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[18],q[15];
RZZ(0.5*pi) q[22],q[30];
RZZ(0.5*pi) q[24],q[23];
RZZ(0.5*pi) q[25],q[27];
RZZ(0.5*pi) q[35],q[28];
RZZ(0.5*pi) q[29],q[36];
RZZ(0.5*pi) q[34],q[33];
U1q(0.496073920899165*pi,1.3056030358813304*pi) q[0];
U1q(0.261028605298713*pi,1.0480567783345602*pi) q[1];
U1q(0.24415099295659*pi,0.4405348332254304*pi) q[2];
U1q(0.393464238335655*pi,1.5403314243520132*pi) q[3];
U1q(0.651551449594431*pi,1.14033153878475*pi) q[4];
U1q(0.144444481834786*pi,1.9530963889959558*pi) q[5];
U1q(0.611810848123665*pi,1.68514075488135*pi) q[6];
U1q(0.515498303068051*pi,0.8840679971084224*pi) q[7];
U1q(0.0853306634609331*pi,0.7991048053588203*pi) q[8];
U1q(0.205859756645718*pi,1.928949724508124*pi) q[9];
U1q(0.357343222100002*pi,1.5472931287654497*pi) q[10];
U1q(0.598863276788657*pi,0.01617989112780016*pi) q[11];
U1q(0.251229888646406*pi,0.4182336552757402*pi) q[12];
U1q(0.662094384438929*pi,1.3363807067217799*pi) q[13];
U1q(0.598407636086769*pi,1.2338341892809384*pi) q[14];
U1q(0.185307050096536*pi,0.015098666159479635*pi) q[15];
U1q(0.378047004403646*pi,0.6922206355578098*pi) q[16];
U1q(0.486726894182978*pi,1.7283914116268702*pi) q[17];
U1q(0.403976629271771*pi,1.3767940686459683*pi) q[18];
U1q(0.3774495865936*pi,1.9263166631826394*pi) q[19];
U1q(0.596117387946524*pi,0.5997740044372302*pi) q[20];
U1q(0.764337346893158*pi,1.1829411770920197*pi) q[21];
U1q(0.445694304471988*pi,0.19413082739213028*pi) q[22];
U1q(0.300677302888259*pi,0.24286068963697982*pi) q[23];
U1q(0.451055024173529*pi,0.8017395424682192*pi) q[24];
U1q(0.183900534803694*pi,1.4785275219389202*pi) q[25];
U1q(0.299542188758329*pi,1.11351803533071*pi) q[26];
U1q(0.302824164415677*pi,0.1808710330022869*pi) q[27];
U1q(0.680349026126163*pi,1.4881960069654*pi) q[28];
U1q(0.856319391367944*pi,0.09142807716840995*pi) q[29];
U1q(0.0694695203607196*pi,0.61129583284165*pi) q[30];
U1q(0.321197973075316*pi,1.2048194674886599*pi) q[31];
U1q(0.566192527352717*pi,1.4929308162914054*pi) q[32];
U1q(0.284881998813604*pi,0.9506200227391455*pi) q[33];
U1q(0.904357432403459*pi,1.97781103849686*pi) q[34];
U1q(0.527790234139895*pi,1.2317345982441497*pi) q[35];
U1q(0.474858947721123*pi,1.2454180203234806*pi) q[36];
U1q(0.352303958562856*pi,1.1625346959328908*pi) q[37];
U1q(0.30192419649249*pi,1.7757108679944258*pi) q[38];
U1q(0.790979667968353*pi,0.5328409635409201*pi) q[39];
RZZ(0.5*pi) q[0],q[32];
RZZ(0.5*pi) q[20],q[1];
RZZ(0.5*pi) q[13],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[7],q[18];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[12],q[15];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[17],q[27];
RZZ(0.5*pi) q[19],q[28];
RZZ(0.5*pi) q[31],q[21];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[39],q[25];
RZZ(0.5*pi) q[26],q[38];
RZZ(0.5*pi) q[37],q[36];
U1q(0.181507348435674*pi,1.13370683775387*pi) q[0];
U1q(0.42414994201387*pi,0.37827256612511917*pi) q[1];
U1q(0.516773810256407*pi,1.3189824112189008*pi) q[2];
U1q(0.246573997944732*pi,1.5243122273548932*pi) q[3];
U1q(0.0496766555817018*pi,0.48804640811562017*pi) q[4];
U1q(0.401206110272935*pi,1.672976223007705*pi) q[5];
U1q(0.587605341452942*pi,0.0770559807927702*pi) q[6];
U1q(0.677402898220935*pi,1.6034915048982334*pi) q[7];
U1q(0.730013733607025*pi,0.7388782372535001*pi) q[8];
U1q(0.32430562469695*pi,0.175083902582994*pi) q[9];
U1q(0.6459700467438*pi,0.8680337617277898*pi) q[10];
U1q(0.690981579156394*pi,1.45491899803896*pi) q[11];
U1q(0.57801792693811*pi,0.9877763155498798*pi) q[12];
U1q(0.122694739972762*pi,1.8413797625330908*pi) q[13];
U1q(0.164016331694403*pi,0.08725669135506742*pi) q[14];
U1q(0.757368348533055*pi,1.1737023553669008*pi) q[15];
U1q(0.0980048333659971*pi,1.1445897224391803*pi) q[16];
U1q(0.230806615397592*pi,0.9751573510799099*pi) q[17];
U1q(0.0925602889037102*pi,0.7085425667918503*pi) q[18];
U1q(0.210285921925995*pi,0.34072010088599924*pi) q[19];
U1q(0.8011253052488*pi,1.8931515838437392*pi) q[20];
U1q(0.314409425636378*pi,0.9590030131934295*pi) q[21];
U1q(0.455836378483453*pi,1.6887439693726005*pi) q[22];
U1q(0.809814797712884*pi,0.18015551503853988*pi) q[23];
U1q(0.286819807626801*pi,0.7878872386738003*pi) q[24];
U1q(0.440353805247344*pi,0.16638227109037018*pi) q[25];
U1q(0.868015403696907*pi,1.02867931633131*pi) q[26];
U1q(0.901950737102548*pi,0.6890105609375077*pi) q[27];
U1q(0.651622177428754*pi,0.10377693635418961*pi) q[28];
U1q(0.325140289008811*pi,0.5194648068563401*pi) q[29];
U1q(0.448960073233789*pi,0.6029774435309996*pi) q[30];
U1q(0.263214529707461*pi,0.6402447088315597*pi) q[31];
U1q(0.417788834838352*pi,1.256034074703905*pi) q[32];
U1q(0.512037778313077*pi,0.671350691664685*pi) q[33];
U1q(0.70024495176747*pi,0.18286901736857075*pi) q[34];
U1q(0.35856619802225*pi,0.08361708612318974*pi) q[35];
U1q(0.450382488791118*pi,0.06288948814649942*pi) q[36];
U1q(0.580749891082103*pi,0.2619923220651703*pi) q[37];
U1q(0.925061346794114*pi,0.5708156969802962*pi) q[38];
U1q(0.404198508942674*pi,1.9807086403334093*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[12],q[2];
RZZ(0.5*pi) q[3],q[10];
RZZ(0.5*pi) q[29],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[13],q[32];
RZZ(0.5*pi) q[14],q[17];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[22],q[38];
RZZ(0.5*pi) q[39],q[23];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[31],q[26];
U1q(0.445318467807956*pi,0.1858455841601998*pi) q[0];
U1q(0.331376729154832*pi,1.1949749904275002*pi) q[1];
U1q(0.736523514182304*pi,0.4408573412148993*pi) q[2];
U1q(0.480184198094303*pi,0.37743868721364393*pi) q[3];
U1q(0.139524685217997*pi,0.6328183526933007*pi) q[4];
U1q(0.623654885971978*pi,1.1844254458630274*pi) q[5];
U1q(0.587517551838203*pi,1.6387488365888991*pi) q[6];
U1q(0.327958347554154*pi,0.1389723925288333*pi) q[7];
U1q(0.631620593557974*pi,1.7459624223947703*pi) q[8];
U1q(0.470210906426249*pi,1.7340209792761936*pi) q[9];
U1q(0.181736722006098*pi,0.21736629079540037*pi) q[10];
U1q(0.455637060533327*pi,0.22109056163144025*pi) q[11];
U1q(0.768099487576998*pi,1.6070860531233997*pi) q[12];
U1q(0.232065350287393*pi,1.7108360152311999*pi) q[13];
U1q(0.703880633534912*pi,1.269925203967567*pi) q[14];
U1q(0.335685548533707*pi,0.4102427778807005*pi) q[15];
U1q(0.614521533451985*pi,0.18290299690159983*pi) q[16];
U1q(0.47501127380753*pi,1.3525187053695902*pi) q[17];
U1q(0.484755231420163*pi,0.4882574491326199*pi) q[18];
U1q(0.0621102811605985*pi,1.0946158558353005*pi) q[19];
U1q(0.26042380604152*pi,1.8933467767808008*pi) q[20];
U1q(0.606967895347681*pi,1.946679424968*pi) q[21];
U1q(0.7386106373205*pi,0.2979110384925896*pi) q[22];
U1q(0.123678267659655*pi,0.9185458584760902*pi) q[23];
U1q(0.433340568669548*pi,1.9080403606304994*pi) q[24];
U1q(0.547811205315595*pi,1.5749848025524997*pi) q[25];
U1q(0.324756281488089*pi,0.49239675934150995*pi) q[26];
U1q(0.761404322232181*pi,0.8001998193085065*pi) q[27];
U1q(0.533635628131076*pi,1.6664948279575995*pi) q[28];
U1q(0.0549590536871169*pi,0.29702713797180014*pi) q[29];
U1q(0.22685003726361*pi,0.08553441234649917*pi) q[30];
U1q(0.666207033922714*pi,1.0445512437061595*pi) q[31];
U1q(0.596004039731104*pi,1.7005567360998057*pi) q[32];
U1q(0.598227787235746*pi,1.2965130594269958*pi) q[33];
U1q(0.368124525907415*pi,1.1269068175063008*pi) q[34];
U1q(0.721310350675132*pi,1.8253828327679997*pi) q[35];
U1q(0.943090648798099*pi,0.6505985385672002*pi) q[36];
U1q(0.228858734293541*pi,0.9091399527543107*pi) q[37];
U1q(0.579206680678807*pi,0.9453995115310843*pi) q[38];
U1q(0.546194106127852*pi,1.6761433740765792*pi) q[39];
RZZ(0.5*pi) q[1],q[0];
RZZ(0.5*pi) q[35],q[2];
RZZ(0.5*pi) q[3],q[38];
RZZ(0.5*pi) q[4],q[39];
RZZ(0.5*pi) q[34],q[5];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[25],q[8];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[12],q[18];
RZZ(0.5*pi) q[22],q[13];
RZZ(0.5*pi) q[14],q[32];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[19],q[23];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[26],q[36];
RZZ(0.5*pi) q[27],q[28];
RZZ(0.5*pi) q[30],q[37];
RZZ(0.5*pi) q[31],q[33];
U1q(0.142398210815114*pi,0.11736659886070022*pi) q[0];
U1q(0.460313510070491*pi,0.35808760936579986*pi) q[1];
U1q(0.246357590508923*pi,0.1464352373442992*pi) q[2];
U1q(0.525979398007446*pi,0.22317444142272258*pi) q[3];
U1q(0.864422271111443*pi,1.2983045142115994*pi) q[4];
U1q(0.294028744449072*pi,0.7433722865752266*pi) q[5];
U1q(0.347303892520382*pi,1.2957631194803998*pi) q[6];
U1q(0.572358170168109*pi,0.8072872269299332*pi) q[7];
U1q(0.51428226433099*pi,1.0709254776044403*pi) q[8];
U1q(0.684234960669637*pi,1.534056609998295*pi) q[9];
U1q(0.218172549972409*pi,0.7666186857241009*pi) q[10];
U1q(0.29347665566346*pi,0.8557661870212598*pi) q[11];
U1q(0.301281266754359*pi,0.5697268928661998*pi) q[12];
U1q(0.272897994954843*pi,0.6566375384393996*pi) q[13];
U1q(0.492641950450399*pi,1.9022539076911666*pi) q[14];
U1q(0.937612953606899*pi,1.8168756954692*pi) q[15];
U1q(0.257382653336821*pi,0.23687533061630006*pi) q[16];
U1q(0.825789814543891*pi,1.5375091399262502*pi) q[17];
U1q(0.286094069536268*pi,1.5773425647354191*pi) q[18];
U1q(0.617218445368531*pi,1.1823906741339005*pi) q[19];
U1q(0.640729889359681*pi,0.5045113317501002*pi) q[20];
U1q(0.676620678901944*pi,1.0552878529340006*pi) q[21];
U1q(0.299279324710082*pi,1.6362997951490001*pi) q[22];
U1q(0.749299633096634*pi,1.4880518099966995*pi) q[23];
U1q(0.153208523185573*pi,1.6739184687966002*pi) q[24];
U1q(0.56284258482924*pi,1.5827124007020998*pi) q[25];
U1q(0.755587776586707*pi,1.3087815308039108*pi) q[26];
U1q(0.0175566814799538*pi,1.273240170455006*pi) q[27];
U1q(0.150035306743852*pi,1.2416352472052008*pi) q[28];
U1q(0.0877160425225174*pi,1.4152410377958002*pi) q[29];
U1q(0.435180342582871*pi,1.5334833925676001*pi) q[30];
U1q(0.860448945511776*pi,1.3085578726158573*pi) q[31];
U1q(0.531358948380858*pi,0.8673696487806755*pi) q[32];
U1q(0.322466717176941*pi,1.3804084523211948*pi) q[33];
U1q(0.52587461523174*pi,1.4907382376112999*pi) q[34];
U1q(0.415897499250322*pi,1.7729073877363994*pi) q[35];
U1q(0.932339423382394*pi,0.24686914267529936*pi) q[36];
U1q(0.226956579627815*pi,1.1274723443785106*pi) q[37];
U1q(0.317891446005245*pi,0.5216223132696758*pi) q[38];
U1q(0.208667600313248*pi,1.3199742115898*pi) q[39];
rz(1.3364570599646992*pi) q[0];
rz(1.5966226939594002*pi) q[1];
rz(2.5864412076990995*pi) q[2];
rz(0.26185917571187645*pi) q[3];
rz(1.6593812837857005*pi) q[4];
rz(3.343664544512272*pi) q[5];
rz(2.6683900121033*pi) q[6];
rz(0.05037008930996478*pi) q[7];
rz(3.78196474949085*pi) q[8];
rz(0.7080286413397054*pi) q[9];
rz(0.9567167708515001*pi) q[10];
rz(2.3130799081824005*pi) q[11];
rz(0.3036478421577016*pi) q[12];
rz(1.9332160646694998*pi) q[13];
rz(0.9184410414835327*pi) q[14];
rz(2.136841507939099*pi) q[15];
rz(2.9445181932414*pi) q[16];
rz(1.3512444543746103*pi) q[17];
rz(2.688861110811681*pi) q[18];
rz(2.9353196283373*pi) q[19];
rz(1.6209600516757003*pi) q[20];
rz(1.2237453670874991*pi) q[21];
rz(0.07998639999190083*pi) q[22];
rz(3.875965694993001*pi) q[23];
rz(3.3262110105161007*pi) q[24];
rz(1.1995199336089009*pi) q[25];
rz(2.46415533723429*pi) q[26];
rz(1.6448233837360942*pi) q[27];
rz(2.8266583779286005*pi) q[28];
rz(1.4853051502679193*pi) q[29];
rz(2.2317904524356003*pi) q[30];
rz(3.61108753065564*pi) q[31];
rz(3.2822658577407235*pi) q[32];
rz(1.9025657143743047*pi) q[33];
rz(2.1028013619795995*pi) q[34];
rz(1.4075521098185995*pi) q[35];
rz(0.18325571679159935*pi) q[36];
rz(3.774956910262789*pi) q[37];
rz(3.6126018439352254*pi) q[38];
rz(3.8043537799623*pi) q[39];
U1q(0.142398210815114*pi,0.453823658825429*pi) q[0];
U1q(0.460313510070491*pi,0.954710303325218*pi) q[1];
U1q(1.24635759050892*pi,1.732876445043486*pi) q[2];
U1q(0.525979398007446*pi,1.485033617134623*pi) q[3];
U1q(0.864422271111443*pi,1.957685797997291*pi) q[4];
U1q(3.294028744449073*pi,1.087036831087456*pi) q[5];
U1q(1.34730389252038*pi,0.9641531315837*pi) q[6];
U1q(0.572358170168109*pi,1.85765731623995*pi) q[7];
U1q(1.51428226433099*pi,1.852890227095284*pi) q[8];
U1q(3.684234960669637*pi,1.24208525133791*pi) q[9];
U1q(1.21817254997241*pi,0.723335456575655*pi) q[10];
U1q(0.29347665566346*pi,0.16884609520368*pi) q[11];
U1q(0.301281266754359*pi,1.873374735023931*pi) q[12];
U1q(0.272897994954843*pi,1.589853603108895*pi) q[13];
U1q(0.492641950450399*pi,1.820694949174746*pi) q[14];
U1q(3.9376129536069*pi,0.953717203408282*pi) q[15];
U1q(0.257382653336821*pi,0.181393523857746*pi) q[16];
U1q(1.82578981454389*pi,1.888753594300863*pi) q[17];
U1q(0.286094069536268*pi,1.266203675547106*pi) q[18];
U1q(0.617218445368531*pi,1.117710302471111*pi) q[19];
U1q(1.64072988935968*pi,1.12547138342579*pi) q[20];
U1q(0.676620678901944*pi,1.2790332200215189*pi) q[21];
U1q(0.299279324710082*pi,0.716286195140857*pi) q[22];
U1q(1.74929963309663*pi,0.364017504989682*pi) q[23];
U1q(1.15320852318557*pi,0.000129479312696246*pi) q[24];
U1q(0.56284258482924*pi,1.782232334310998*pi) q[25];
U1q(0.755587776586707*pi,0.772936868038173*pi) q[26];
U1q(1.01755668147995*pi,1.9180635541911064*pi) q[27];
U1q(1.15003530674385*pi,1.068293625133885*pi) q[28];
U1q(0.0877160425225174*pi,1.9005461880637182*pi) q[29];
U1q(0.435180342582871*pi,0.765273845003155*pi) q[30];
U1q(0.860448945511776*pi,1.9196454032715549*pi) q[31];
U1q(1.53135894838086*pi,1.1496355065214852*pi) q[32];
U1q(3.322466717176941*pi,0.282974166695408*pi) q[33];
U1q(0.52587461523174*pi,0.593539599590981*pi) q[34];
U1q(0.415897499250322*pi,0.180459497554974*pi) q[35];
U1q(0.932339423382394*pi,1.4301248594668858*pi) q[36];
U1q(3.226956579627815*pi,1.9024292546413328*pi) q[37];
U1q(1.31789144600525*pi,1.134224157204871*pi) q[38];
U1q(0.208667600313248*pi,0.124327991552161*pi) q[39];
RZZ(0.5*pi) q[1],q[0];
RZZ(0.5*pi) q[35],q[2];
RZZ(0.5*pi) q[3],q[38];
RZZ(0.5*pi) q[4],q[39];
RZZ(0.5*pi) q[34],q[5];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[25],q[8];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[12],q[18];
RZZ(0.5*pi) q[22],q[13];
RZZ(0.5*pi) q[14],q[32];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[19],q[23];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[26],q[36];
RZZ(0.5*pi) q[27],q[28];
RZZ(0.5*pi) q[30],q[37];
RZZ(0.5*pi) q[31],q[33];
U1q(1.44531846780796*pi,1.522302644124925*pi) q[0];
U1q(0.331376729154832*pi,0.7915976843869299*pi) q[1];
U1q(1.7365235141823*pi,1.438454341172969*pi) q[2];
U1q(0.480184198094303*pi,1.63929786292556*pi) q[3];
U1q(0.139524685217997*pi,0.29219963647902003*pi) q[4];
U1q(3.623654885971978*pi,0.645983671799701*pi) q[5];
U1q(1.5875175518382*pi,0.621167414475251*pi) q[6];
U1q(1.32795834755415*pi,1.189342481838791*pi) q[7];
U1q(3.368379406442026*pi,0.17785328230495034*pi) q[8];
U1q(3.529789093573751*pi,0.04212088205996389*pi) q[9];
U1q(1.1817367220061*pi,1.272587851504392*pi) q[10];
U1q(1.45563706053333*pi,1.534170469813861*pi) q[11];
U1q(1.768099487577*pi,1.9107338952811102*pi) q[12];
U1q(0.232065350287393*pi,1.64405207990069*pi) q[13];
U1q(0.703880633534912*pi,1.188366245451196*pi) q[14];
U1q(3.664314451466293*pi,0.36035012099671576*pi) q[15];
U1q(0.614521533451985*pi,0.12742119014300002*pi) q[16];
U1q(1.47501127380753*pi,0.0737440288575232*pi) q[17];
U1q(0.484755231420163*pi,1.1771185599443301*pi) q[18];
U1q(0.0621102811605985*pi,1.029935484172589*pi) q[19];
U1q(1.26042380604152*pi,1.7366359383951033*pi) q[20];
U1q(0.606967895347681*pi,0.17042479205553995*pi) q[21];
U1q(1.7386106373205*pi,0.377897438484486*pi) q[22];
U1q(1.12367826765966*pi,0.9335234565103069*pi) q[23];
U1q(1.43334056866955*pi,0.7660075874787897*pi) q[24];
U1q(0.547811205315595*pi,1.774504736161417*pi) q[25];
U1q(1.32475628148809*pi,0.956552096575792*pi) q[26];
U1q(1.76140432223218*pi,1.3911039053376288*pi) q[27];
U1q(1.53363562813108*pi,1.6434340443815483*pi) q[28];
U1q(0.0549590536871169*pi,0.7823322882397199*pi) q[29];
U1q(0.22685003726361*pi,1.317324864782079*pi) q[30];
U1q(0.666207033922714*pi,1.6556387743618401*pi) q[31];
U1q(3.403995960268896*pi,1.3164484192024002*pi) q[32];
U1q(1.59822778723575*pi,0.3668695595895357*pi) q[33];
U1q(0.368124525907415*pi,0.2297081794859701*pi) q[34];
U1q(0.721310350675132*pi,0.232934942586508*pi) q[35];
U1q(0.943090648798099*pi,0.8338542553588*pi) q[36];
U1q(3.228858734293541*pi,0.12076164626550101*pi) q[37];
U1q(3.4207933193211932*pi,1.7104469589434161*pi) q[38];
U1q(0.546194106127852*pi,1.480497154038913*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[12],q[2];
RZZ(0.5*pi) q[3],q[10];
RZZ(0.5*pi) q[29],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[13],q[32];
RZZ(0.5*pi) q[14],q[17];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[22],q[38];
RZZ(0.5*pi) q[39],q[23];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[31],q[26];
U1q(1.18150734843567*pi,0.5744413905312535*pi) q[0];
U1q(1.42414994201387*pi,0.97489526008453*pi) q[1];
U1q(0.516773810256407*pi,0.3165794111769761*pi) q[2];
U1q(3.246573997944731*pi,1.7861714030668199*pi) q[3];
U1q(0.0496766555817018*pi,1.14742769190135*pi) q[4];
U1q(0.401206110272935*pi,0.13453444894442512*pi) q[5];
U1q(3.587605341452941*pi,0.05947455867916096*pi) q[6];
U1q(3.322597101779065*pi,0.7248233694693385*pi) q[7];
U1q(1.73001373360703*pi,0.1849374674462233*pi) q[8];
U1q(3.67569437530305*pi,1.601057958753213*pi) q[9];
U1q(3.6459700467438*pi,0.923255322436781*pi) q[10];
U1q(1.69098157915639*pi,0.3003420334063473*pi) q[11];
U1q(1.57801792693811*pi,0.5300436328546612*pi) q[12];
U1q(0.122694739972762*pi,1.7745958272025701*pi) q[13];
U1q(0.164016331694403*pi,1.0056977328386498*pi) q[14];
U1q(3.242631651466945*pi,1.5968905435105936*pi) q[15];
U1q(3.098004833365997*pi,0.08910791568058007*pi) q[16];
U1q(3.230806615397592*pi,0.6963826745678403*pi) q[17];
U1q(0.0925602889037102*pi,0.3974036776035099*pi) q[18];
U1q(3.210285921925995*pi,0.2760397292232599*pi) q[19];
U1q(0.8011253052488*pi,1.736440745458053*pi) q[20];
U1q(0.314409425636378*pi,0.18274838028092022*pi) q[21];
U1q(1.45583637848345*pi,1.9870645076044804*pi) q[22];
U1q(0.809814797712884*pi,1.195133113072755*pi) q[23];
U1q(3.286819807626801*pi,0.6458544655220737*pi) q[24];
U1q(1.44035380524734*pi,0.3659022046992799*pi) q[25];
U1q(1.86801540369691*pi,0.42026953958599045*pi) q[26];
U1q(1.90195073710255*pi,0.2799146469666487*pi) q[27];
U1q(0.651622177428754*pi,0.08071615277815347*pi) q[28];
U1q(1.32514028900881*pi,0.0047699571242598715*pi) q[29];
U1q(0.448960073233789*pi,0.8347678959665901*pi) q[30];
U1q(1.26321452970746*pi,1.25133223948718*pi) q[31];
U1q(1.41778883483835*pi,0.7609710805982973*pi) q[32];
U1q(0.512037778313077*pi,0.7417071918272037*pi) q[33];
U1q(0.70024495176747*pi,1.28567037934822*pi) q[34];
U1q(1.35856619802225*pi,1.49116919594174*pi) q[35];
U1q(0.450382488791118*pi,1.24614520493807*pi) q[36];
U1q(0.580749891082103*pi,1.473614015576341*pi) q[37];
U1q(3.074938653205886*pi,1.0850307734942002*pi) q[38];
U1q(0.404198508942674*pi,0.7850624202957399*pi) q[39];
RZZ(0.5*pi) q[0],q[32];
RZZ(0.5*pi) q[20],q[1];
RZZ(0.5*pi) q[13],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[7],q[18];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[12],q[15];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[17],q[27];
RZZ(0.5*pi) q[19],q[28];
RZZ(0.5*pi) q[31],q[21];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[39],q[25];
RZZ(0.5*pi) q[26],q[38];
RZZ(0.5*pi) q[37],q[36];
U1q(0.496073920899165*pi,1.7463375886587085*pi) q[0];
U1q(3.738971394701288*pi,1.3051110478750938*pi) q[1];
U1q(1.24415099295659*pi,0.4381318331835429*pi) q[2];
U1q(3.393464238335655*pi,1.7701522060697084*pi) q[3];
U1q(3.651551449594431*pi,0.7997128225704904*pi) q[4];
U1q(0.144444481834786*pi,1.414654614932675*pi) q[5];
U1q(1.61181084812367*pi,0.45138978459058854*pi) q[6];
U1q(1.51549830306805*pi,0.4442468772591981*pi) q[7];
U1q(0.0853306634609331*pi,0.2451640355515512*pi) q[8];
U1q(3.794140243354282*pi,1.84719213682805*pi) q[9];
U1q(3.642656777899998*pi,1.2439959553991229*pi) q[10];
U1q(0.598863276788657*pi,1.8616029264951863*pi) q[11];
U1q(0.251229888646406*pi,1.9605009725805314*pi) q[12];
U1q(1.66209438443893*pi,0.26959677139126015*pi) q[13];
U1q(0.598407636086769*pi,0.15227523076451988*pi) q[14];
U1q(3.814692949903463*pi,0.7554942327179739*pi) q[15];
U1q(3.621952995596355*pi,1.541477002561951*pi) q[16];
U1q(1.48672689418298*pi,0.9431486140208714*pi) q[17];
U1q(0.403976629271771*pi,1.0656551794576297*pi) q[18];
U1q(1.3774495865936*pi,0.6904431669266196*pi) q[19];
U1q(0.596117387946524*pi,0.44306316605154317*pi) q[20];
U1q(1.76433734689316*pi,0.40668654417951*pi) q[21];
U1q(1.44569430447199*pi,0.4924513656240146*pi) q[22];
U1q(1.30067730288826*pi,1.2578382876711949*pi) q[23];
U1q(3.54894497582647*pi,0.6320021617276401*pi) q[24];
U1q(1.18390053480369*pi,1.0537569538507314*pi) q[25];
U1q(1.29954218875833*pi,0.5051082585854085*pi) q[26];
U1q(3.697175835584323*pi,0.7880541749018832*pi) q[27];
U1q(0.680349026126163*pi,1.465135223389363*pi) q[28];
U1q(1.85631939136794*pi,0.4328066868121958*pi) q[29];
U1q(1.06946952036072*pi,0.8430862852772201*pi) q[30];
U1q(3.678802026924684*pi,1.6867574808300394*pi) q[31];
U1q(0.566192527352717*pi,1.9978678221857993*pi) q[32];
U1q(0.284881998813604*pi,0.020976522901657857*pi) q[33];
U1q(0.904357432403459*pi,1.08061240047651*pi) q[34];
U1q(3.472209765860105*pi,0.343051683820788*pi) q[35];
U1q(0.474858947721123*pi,1.4286737371150604*pi) q[36];
U1q(1.35230395856286*pi,1.3741563894440585*pi) q[37];
U1q(3.6980758035075088*pi,1.88013560248007*pi) q[38];
U1q(0.790979667968353*pi,1.33719474350326*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[20],q[2];
RZZ(0.5*pi) q[4],q[37];
RZZ(0.5*pi) q[5],q[32];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[9],q[26];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[12],q[21];
RZZ(0.5*pi) q[17],q[13];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[18],q[15];
RZZ(0.5*pi) q[22],q[30];
RZZ(0.5*pi) q[24],q[23];
RZZ(0.5*pi) q[25],q[27];
RZZ(0.5*pi) q[35],q[28];
RZZ(0.5*pi) q[29],q[36];
RZZ(0.5*pi) q[34],q[33];
U1q(1.87935796920841*pi,1.730695302300889*pi) q[0];
U1q(3.046181963836581*pi,1.0315208100612736*pi) q[1];
U1q(3.39946105184518*pi,1.1617640759319758*pi) q[2];
U1q(3.644946876409075*pi,1.8107662301064784*pi) q[3];
U1q(3.732490561188639*pi,1.0747904906244177*pi) q[4];
U1q(0.601224290219005*pi,1.185910596806055*pi) q[5];
U1q(0.444893486897357*pi,1.1076568164701586*pi) q[6];
U1q(3.482328190782586*pi,0.8084654372351281*pi) q[7];
U1q(1.6221233939587*pi,0.3396389352591913*pi) q[8];
U1q(1.62368882841584*pi,0.8344564388893887*pi) q[9];
U1q(1.67761083761569*pi,0.917716636371235*pi) q[10];
U1q(0.804901502544758*pi,1.4424011042494764*pi) q[11];
U1q(0.366055391870163*pi,0.11434076447984065*pi) q[12];
U1q(3.614002138089934*pi,1.1030734921246026*pi) q[13];
U1q(0.33578436202363*pi,0.11071939282270993*pi) q[14];
U1q(3.258100167438163*pi,0.19227471461739398*pi) q[15];
U1q(3.707044844032274*pi,1.6359947894106686*pi) q[16];
U1q(1.64170441740492*pi,1.8593485364912543*pi) q[17];
U1q(0.0884609881801889*pi,1.2312074994382396*pi) q[18];
U1q(3.331469517919423*pi,1.2143320074948996*pi) q[19];
U1q(0.513107240936898*pi,0.19997193396198343*pi) q[20];
U1q(1.7335202017931*pi,1.5723348459664184*pi) q[21];
U1q(1.53550618655701*pi,1.3031770141258487*pi) q[22];
U1q(1.6986884974124*pi,0.861993102424218*pi) q[23];
U1q(1.83868847798604*pi,0.6335314093891906*pi) q[24];
U1q(0.652162429422769*pi,0.8239707437625314*pi) q[25];
U1q(3.420450844626819*pi,1.710109992269515*pi) q[26];
U1q(1.84490270704022*pi,1.125665514513872*pi) q[27];
U1q(0.14575248303595*pi,0.49556749871802364*pi) q[28];
U1q(0.534114791604311*pi,0.13456677582975596*pi) q[29];
U1q(3.422576704867057*pi,0.09531659884348476*pi) q[30];
U1q(1.43796895766441*pi,0.9281078150846618*pi) q[31];
U1q(1.60153119948811*pi,1.4581698439120796*pi) q[32];
U1q(0.627116011480857*pi,0.32753121668498775*pi) q[33];
U1q(1.49387589494394*pi,1.1291275389857298*pi) q[34];
U1q(1.61914125601556*pi,1.1275650255410454*pi) q[35];
U1q(1.14296147999057*pi,0.6959136547126903*pi) q[36];
U1q(3.309537838788963*pi,1.779106810694011*pi) q[37];
U1q(3.2377819408039112*pi,1.0990558635079102*pi) q[38];
U1q(1.4043268768214*pi,1.3036735686753804*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[3],q[4];
RZZ(0.5*pi) q[20],q[5];
RZZ(0.5*pi) q[6],q[35];
RZZ(0.5*pi) q[22],q[7];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[10],q[32];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[12],q[34];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[26],q[15];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[38],q[23];
RZZ(0.5*pi) q[29],q[24];
RZZ(0.5*pi) q[25],q[36];
RZZ(0.5*pi) q[39],q[33];
U1q(3.335911659108376*pi,1.2787379967657309*pi) q[0];
U1q(1.60271166981823*pi,1.0665481022558478*pi) q[1];
U1q(3.816611924972754*pi,1.3535593972794957*pi) q[2];
U1q(3.0254344055550693*pi,1.2143658398406516*pi) q[3];
U1q(1.16683401234745*pi,0.7810348351866789*pi) q[4];
U1q(1.1526308629148*pi,1.6558988359099454*pi) q[5];
U1q(0.539200773277447*pi,1.1729755241765587*pi) q[6];
U1q(3.098101541478988*pi,1.626365565865136*pi) q[7];
U1q(3.718097984220955*pi,1.250141153140909*pi) q[8];
U1q(0.523763207869831*pi,0.11973084048563765*pi) q[9];
U1q(0.377693307328995*pi,0.20500771985148614*pi) q[10];
U1q(3.827051950085107*pi,0.11526806240011023*pi) q[11];
U1q(1.56697785953332*pi,1.268542596780561*pi) q[12];
U1q(3.072184769299684*pi,1.2381169923026327*pi) q[13];
U1q(1.34124786657547*pi,1.6469295924914604*pi) q[14];
U1q(3.3516796999694582*pi,1.236995266442884*pi) q[15];
U1q(0.426899309430264*pi,0.8695681561599784*pi) q[16];
U1q(1.73466143010458*pi,0.13596132708545206*pi) q[17];
U1q(1.15344294931327*pi,0.7369029465204102*pi) q[18];
U1q(3.528226487399621*pi,0.10264945171265083*pi) q[19];
U1q(1.11763376917564*pi,0.0654595604602135*pi) q[20];
U1q(0.711441447251137*pi,0.11212758963751845*pi) q[21];
U1q(0.66278702211824*pi,0.9010471717219684*pi) q[22];
U1q(0.830079556254018*pi,1.3687431545756779*pi) q[23];
U1q(1.88646322240395*pi,1.7335637016189906*pi) q[24];
U1q(0.619119957674733*pi,0.2618969649420515*pi) q[25];
U1q(1.05944562357664*pi,1.7734327596746071*pi) q[26];
U1q(0.70607011768905*pi,0.577212907723422*pi) q[27];
U1q(0.70713865915301*pi,0.7017231597618636*pi) q[28];
U1q(0.730212891504842*pi,1.4085577414281865*pi) q[29];
U1q(3.6478720822433592*pi,1.2530003361867088*pi) q[30];
U1q(1.320269116938*pi,1.4995904095318116*pi) q[31];
U1q(3.766258093144708*pi,0.006723240851012946*pi) q[32];
U1q(1.24445772750932*pi,0.6877649155171177*pi) q[33];
U1q(1.59484673339105*pi,1.8775324382296472*pi) q[34];
U1q(1.39891301569412*pi,0.34689968158299544*pi) q[35];
U1q(1.32540595583092*pi,1.1804267612028196*pi) q[36];
U1q(1.47997796949463*pi,0.4566357965629475*pi) q[37];
U1q(1.55995106788034*pi,0.8600289790399975*pi) q[38];
U1q(3.3470513242574*pi,0.6526811468429372*pi) q[39];
RZZ(0.5*pi) q[29],q[0];
RZZ(0.5*pi) q[35],q[1];
RZZ(0.5*pi) q[26],q[2];
RZZ(0.5*pi) q[3],q[17];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[12],q[6];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[38];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[14],q[18];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[16],q[28];
RZZ(0.5*pi) q[19],q[21];
RZZ(0.5*pi) q[20],q[39];
RZZ(0.5*pi) q[22],q[24];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[25],q[37];
U1q(0.40930381121135*pi,0.8003752480901811*pi) q[0];
U1q(0.512259288312356*pi,1.0400475534219376*pi) q[1];
U1q(1.07266776977313*pi,0.19482704934309636*pi) q[2];
U1q(1.69397670785088*pi,0.12253548374172851*pi) q[3];
U1q(0.217347225213229*pi,1.3396740315010796*pi) q[4];
U1q(3.733644371285739*pi,0.5304947149741421*pi) q[5];
U1q(0.877657894072522*pi,1.3021682314635228*pi) q[6];
U1q(1.8141422073031*pi,0.3870713901532348*pi) q[7];
U1q(1.87512887387104*pi,1.0870157625893988*pi) q[8];
U1q(0.483173838904506*pi,0.2086413579771037*pi) q[9];
U1q(0.604782902755503*pi,0.026704020226718056*pi) q[10];
U1q(1.75194519417096*pi,0.008565354119503432*pi) q[11];
U1q(3.628062213644457*pi,0.32457032221872595*pi) q[12];
U1q(1.25746875230635*pi,1.533584577344656*pi) q[13];
U1q(3.537914578007414*pi,0.044375836818074*pi) q[14];
U1q(1.02472780419062*pi,1.2712955276860414*pi) q[15];
U1q(0.658632000997557*pi,0.662492004256559*pi) q[16];
U1q(0.613385602594099*pi,1.529697570200672*pi) q[17];
U1q(1.28230725789889*pi,1.1503940816338334*pi) q[18];
U1q(0.198764731253661*pi,0.33703564285949117*pi) q[19];
U1q(1.60464409326375*pi,0.8995555148744501*pi) q[20];
U1q(0.92774843136428*pi,1.6458597837860482*pi) q[21];
U1q(0.169700944991006*pi,1.4528881577194488*pi) q[22];
U1q(0.410841022524549*pi,1.3385430870296178*pi) q[23];
U1q(3.364714896052908*pi,1.5932631761212726*pi) q[24];
U1q(0.231689779809899*pi,0.02717612849247164*pi) q[25];
U1q(0.612968091816546*pi,0.2372715323292982*pi) q[26];
U1q(0.482635411548189*pi,1.5977978633546819*pi) q[27];
U1q(0.599237093574775*pi,0.4821187287461237*pi) q[28];
U1q(0.696510930820612*pi,1.9306587410464662*pi) q[29];
U1q(0.613251084095671*pi,0.9774855079728857*pi) q[30];
U1q(1.42651259957532*pi,0.15834951971124678*pi) q[31];
U1q(3.405149512875348*pi,1.149109014991998*pi) q[32];
U1q(1.73064435578518*pi,1.6364047682226266*pi) q[33];
U1q(0.550254184137239*pi,0.10637238936454718*pi) q[34];
U1q(1.42018826103322*pi,1.5702809402963362*pi) q[35];
U1q(0.182672337169625*pi,0.4055555838120002*pi) q[36];
U1q(0.5398638841728*pi,0.5963392207646585*pi) q[37];
U1q(0.688688742203591*pi,1.4036883007334682*pi) q[38];
U1q(1.7639944310855*pi,0.017593762345503805*pi) q[39];
rz(3.199624751909819*pi) q[0];
rz(2.9599524465780624*pi) q[1];
rz(3.8051729506569036*pi) q[2];
rz(3.8774645162582715*pi) q[3];
rz(0.6603259684989204*pi) q[4];
rz(1.4695052850258579*pi) q[5];
rz(0.6978317685364772*pi) q[6];
rz(1.6129286098467652*pi) q[7];
rz(0.9129842374106012*pi) q[8];
rz(3.7913586420228964*pi) q[9];
rz(3.973295979773282*pi) q[10];
rz(1.9914346458804966*pi) q[11];
rz(1.675429677781274*pi) q[12];
rz(0.4664154226553441*pi) q[13];
rz(1.955624163181926*pi) q[14];
rz(2.7287044723139586*pi) q[15];
rz(3.337507995743441*pi) q[16];
rz(0.4703024297993279*pi) q[17];
rz(2.8496059183661666*pi) q[18];
rz(3.662964357140509*pi) q[19];
rz(1.10044448512555*pi) q[20];
rz(2.3541402162139518*pi) q[21];
rz(0.5471118422805512*pi) q[22];
rz(2.6614569129703822*pi) q[23];
rz(0.40673682387872745*pi) q[24];
rz(3.9728238715075284*pi) q[25];
rz(1.7627284676707018*pi) q[26];
rz(2.402202136645318*pi) q[27];
rz(3.5178812712538763*pi) q[28];
rz(2.069341258953534*pi) q[29];
rz(3.0225144920271143*pi) q[30];
rz(3.8416504802887532*pi) q[31];
rz(0.850890985008002*pi) q[32];
rz(2.3635952317773734*pi) q[33];
rz(3.893627610635453*pi) q[34];
rz(0.42971905970366375*pi) q[35];
rz(3.594444416188*pi) q[36];
rz(3.4036607792353415*pi) q[37];
rz(2.596311699266532*pi) q[38];
rz(3.982406237654496*pi) q[39];
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
