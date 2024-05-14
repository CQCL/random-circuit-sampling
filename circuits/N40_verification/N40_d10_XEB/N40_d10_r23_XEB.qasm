OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.69657738765043*pi,1.36817523889804*pi) q[0];
U1q(0.289175464803891*pi,1.9128925041222955*pi) q[1];
U1q(0.436456587850923*pi,0.360229270580513*pi) q[2];
U1q(0.630913300440157*pi,0.666467038696858*pi) q[3];
U1q(0.741202824351968*pi,1.5131573950057269*pi) q[4];
U1q(0.436065381587895*pi,0.9593027040472799*pi) q[5];
U1q(0.166126904821815*pi,1.1404263083475281*pi) q[6];
U1q(0.360581039378208*pi,1.893528900925972*pi) q[7];
U1q(0.393493608177476*pi,1.670561307185799*pi) q[8];
U1q(0.136899250708609*pi,0.038328556495417*pi) q[9];
U1q(0.28886182762803*pi,1.82101402613509*pi) q[10];
U1q(0.284212737318583*pi,0.29673988735585*pi) q[11];
U1q(0.537938531560985*pi,0.514797678607616*pi) q[12];
U1q(0.445353384280993*pi,1.505179745458803*pi) q[13];
U1q(0.867180308880772*pi,0.97389215744225*pi) q[14];
U1q(0.657261525913964*pi,0.0454585759003154*pi) q[15];
U1q(0.181223785632738*pi,1.788209197458439*pi) q[16];
U1q(0.632539824257439*pi,0.241375995039721*pi) q[17];
U1q(0.688692196365196*pi,1.574063241527118*pi) q[18];
U1q(0.338309703412928*pi,0.84645677773168*pi) q[19];
U1q(0.541028462942419*pi,1.05449564730607*pi) q[20];
U1q(0.618694673870478*pi,1.10085031986017*pi) q[21];
U1q(0.607607906148919*pi,0.210302208289765*pi) q[22];
U1q(0.642609288865298*pi,0.0350061416274123*pi) q[23];
U1q(0.435944888723193*pi,1.859992672455735*pi) q[24];
U1q(0.241830686384848*pi,1.577768184614097*pi) q[25];
U1q(0.64189612021952*pi,0.922507381791969*pi) q[26];
U1q(0.515532268888338*pi,1.359208803941192*pi) q[27];
U1q(0.505685160038639*pi,0.260776406420691*pi) q[28];
U1q(0.75517235128259*pi,1.38048775727111*pi) q[29];
U1q(0.590335041588302*pi,0.874580532031596*pi) q[30];
U1q(0.361118486631084*pi,1.209558052677643*pi) q[31];
U1q(0.543575685456537*pi,1.2045225380686642*pi) q[32];
U1q(0.337163579430138*pi,1.57535876843413*pi) q[33];
U1q(0.547414180142619*pi,1.86621644569905*pi) q[34];
U1q(0.604851814236212*pi,0.756652677988737*pi) q[35];
U1q(0.504737813368112*pi,0.0493320182467968*pi) q[36];
U1q(0.821127306495896*pi,1.864468667993777*pi) q[37];
U1q(0.544163060660349*pi,1.9567226205636226*pi) q[38];
U1q(0.803143810017386*pi,0.411250325082608*pi) q[39];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[1],q[23];
RZZ(0.5*pi) q[2],q[14];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[4],q[37];
RZZ(0.5*pi) q[20],q[5];
RZZ(0.5*pi) q[6],q[22];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[8],q[38];
RZZ(0.5*pi) q[31],q[9];
RZZ(0.5*pi) q[16],q[10];
RZZ(0.5*pi) q[11],q[15];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[19],q[13];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[21],q[24];
RZZ(0.5*pi) q[25],q[27];
RZZ(0.5*pi) q[29],q[39];
RZZ(0.5*pi) q[34],q[30];
RZZ(0.5*pi) q[35],q[33];
U1q(0.720586935973861*pi,0.37970057118553*pi) q[0];
U1q(0.936642937067155*pi,0.9875818017853*pi) q[1];
U1q(0.51791262129899*pi,0.3567004867027801*pi) q[2];
U1q(0.533371452731569*pi,1.77972082308738*pi) q[3];
U1q(0.301950838621907*pi,0.1495623752629398*pi) q[4];
U1q(0.80207999182822*pi,1.2403634264230998*pi) q[5];
U1q(0.619390474888101*pi,0.7437146005014301*pi) q[6];
U1q(0.783302683351351*pi,1.67766176014399*pi) q[7];
U1q(0.122623079601376*pi,1.4899486517446099*pi) q[8];
U1q(0.464083749039049*pi,0.018639093776790183*pi) q[9];
U1q(0.264643344575967*pi,1.2394703728077299*pi) q[10];
U1q(0.46240799740014*pi,1.93251336304751*pi) q[11];
U1q(0.666529977667066*pi,0.6525665287846101*pi) q[12];
U1q(0.456104226879225*pi,1.8244032890614301*pi) q[13];
U1q(0.486876523239143*pi,0.42319990724574996*pi) q[14];
U1q(0.108225810239244*pi,0.8597798372765699*pi) q[15];
U1q(0.668271566042055*pi,1.81821488903915*pi) q[16];
U1q(0.760198203778803*pi,1.0713386342961502*pi) q[17];
U1q(0.248772779373608*pi,1.0069426235981802*pi) q[18];
U1q(0.252545333808404*pi,1.1092673087875302*pi) q[19];
U1q(0.382088432895699*pi,1.764645337251443*pi) q[20];
U1q(0.5771995297273*pi,0.046845916987910075*pi) q[21];
U1q(0.527143612878553*pi,1.5281656225225602*pi) q[22];
U1q(0.186872283692963*pi,0.7979936484196801*pi) q[23];
U1q(0.0486681279152708*pi,0.9271857339863101*pi) q[24];
U1q(0.349994233854802*pi,0.002533015115139925*pi) q[25];
U1q(0.292818224753259*pi,0.8524194136415499*pi) q[26];
U1q(0.510653490600929*pi,0.19004203871857994*pi) q[27];
U1q(0.716391179207395*pi,0.8671375314731802*pi) q[28];
U1q(0.637473320240794*pi,1.312822467845264*pi) q[29];
U1q(0.503088391797848*pi,1.717398637223519*pi) q[30];
U1q(0.531120565154946*pi,0.5926410509326201*pi) q[31];
U1q(0.200670352671039*pi,1.0835082442553898*pi) q[32];
U1q(0.635372594041191*pi,0.7925003313974401*pi) q[33];
U1q(0.724138754858999*pi,1.87653590725867*pi) q[34];
U1q(0.569607553981004*pi,1.8702074632614*pi) q[35];
U1q(0.673568834588901*pi,0.7694424917701601*pi) q[36];
U1q(0.559323288960756*pi,1.2844870272898299*pi) q[37];
U1q(0.763871641898634*pi,1.7519342122764998*pi) q[38];
U1q(0.811808075527973*pi,1.80643226595897*pi) q[39];
RZZ(0.5*pi) q[24],q[0];
RZZ(0.5*pi) q[1],q[37];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[4],q[14];
RZZ(0.5*pi) q[5],q[35];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[8],q[19];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[12],q[13];
RZZ(0.5*pi) q[15],q[36];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[25],q[32];
RZZ(0.5*pi) q[31],q[29];
RZZ(0.5*pi) q[38],q[30];
U1q(0.649192753232086*pi,0.7242839697285199*pi) q[0];
U1q(0.708748320242314*pi,0.5419446000865902*pi) q[1];
U1q(0.0737712175627972*pi,0.7219779758109302*pi) q[2];
U1q(0.53567991678826*pi,1.31835245995555*pi) q[3];
U1q(0.696334545154579*pi,1.46058856963357*pi) q[4];
U1q(0.363831019498561*pi,0.7900464938420999*pi) q[5];
U1q(0.156396190991037*pi,0.4037139408218797*pi) q[6];
U1q(0.73072512471707*pi,1.0534155364862903*pi) q[7];
U1q(0.0743714055613789*pi,0.13015792863791997*pi) q[8];
U1q(0.815456450802024*pi,0.6531397687059597*pi) q[9];
U1q(0.322132636399949*pi,1.9278674331517198*pi) q[10];
U1q(0.460071883023273*pi,0.7628481279710604*pi) q[11];
U1q(0.275688338802647*pi,1.5020800574151298*pi) q[12];
U1q(0.408745395737047*pi,0.6289842521027502*pi) q[13];
U1q(0.204530557619987*pi,1.5630261429117498*pi) q[14];
U1q(0.902089695749153*pi,1.2235699927760502*pi) q[15];
U1q(0.641850431446149*pi,0.0336402130906599*pi) q[16];
U1q(0.575149130010574*pi,1.2918977791921504*pi) q[17];
U1q(0.536442326817136*pi,1.6599301029174596*pi) q[18];
U1q(0.667736838719509*pi,0.5218157501124496*pi) q[19];
U1q(0.391007031674721*pi,1.9099112428937102*pi) q[20];
U1q(0.524215549256337*pi,1.90833773051519*pi) q[21];
U1q(0.673091181561031*pi,1.6335458952193296*pi) q[22];
U1q(0.7523265566309*pi,0.96372702581189*pi) q[23];
U1q(0.74640615422943*pi,1.0292177832395097*pi) q[24];
U1q(0.607807038980432*pi,0.12657649754306988*pi) q[25];
U1q(0.483633893137681*pi,1.6711507736815499*pi) q[26];
U1q(0.407567159172145*pi,0.9997020327274297*pi) q[27];
U1q(0.422764241892196*pi,1.9502997195909*pi) q[28];
U1q(0.524314109369118*pi,0.9349227581213402*pi) q[29];
U1q(0.721919459806697*pi,1.9499645327402102*pi) q[30];
U1q(0.524614461736596*pi,1.5488793897862*pi) q[31];
U1q(0.853359231906341*pi,1.69266051715947*pi) q[32];
U1q(0.854396292180648*pi,0.2618587892662898*pi) q[33];
U1q(0.164853461801978*pi,0.5832546689817102*pi) q[34];
U1q(0.255168999378543*pi,1.4590304740972098*pi) q[35];
U1q(0.543004167937892*pi,0.8438972440867398*pi) q[36];
U1q(0.524015611920714*pi,1.22285503765288*pi) q[37];
U1q(0.74405171252415*pi,1.06303935833765*pi) q[38];
U1q(0.743742582889928*pi,1.4027275053504997*pi) q[39];
RZZ(0.5*pi) q[18],q[0];
RZZ(0.5*pi) q[1],q[9];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[24];
RZZ(0.5*pi) q[4],q[39];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[32],q[8];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[16],q[33];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[25],q[23];
RZZ(0.5*pi) q[31],q[37];
U1q(0.377879330132176*pi,0.2218316945873*pi) q[0];
U1q(0.59648056621311*pi,1.83530585782961*pi) q[1];
U1q(0.679957041293874*pi,1.1427459369452198*pi) q[2];
U1q(0.688885063662498*pi,1.8179423214836703*pi) q[3];
U1q(0.551783149640857*pi,1.21779715458249*pi) q[4];
U1q(0.532403656764512*pi,1.6593687032838993*pi) q[5];
U1q(0.299337198449784*pi,1.2735817716482707*pi) q[6];
U1q(0.73676470379307*pi,1.06184509680969*pi) q[7];
U1q(0.523717574979256*pi,0.9190651608887697*pi) q[8];
U1q(0.129326114589356*pi,0.8981725194018297*pi) q[9];
U1q(0.344660832050264*pi,0.3140344861872899*pi) q[10];
U1q(0.476972174566075*pi,1.6434497059122908*pi) q[11];
U1q(0.624334525601981*pi,0.7038879183635203*pi) q[12];
U1q(0.432748953178116*pi,1.8126484025509395*pi) q[13];
U1q(0.346339847053081*pi,0.8781245484381399*pi) q[14];
U1q(0.234371104713792*pi,0.8246720107783698*pi) q[15];
U1q(0.594760760093006*pi,1.2280054329509102*pi) q[16];
U1q(0.251390378657405*pi,1.1925300020180396*pi) q[17];
U1q(0.547621907548179*pi,1.8912718771036898*pi) q[18];
U1q(0.426068320565621*pi,0.03763854612630002*pi) q[19];
U1q(0.394455854147944*pi,1.6855860451573301*pi) q[20];
U1q(0.61764894283172*pi,0.96496194747324*pi) q[21];
U1q(0.726623824664051*pi,0.047543493321989594*pi) q[22];
U1q(0.799780155185878*pi,1.3214770712116497*pi) q[23];
U1q(0.749868325830334*pi,1.5661819187951602*pi) q[24];
U1q(0.38019982978423*pi,0.24733617983291012*pi) q[25];
U1q(0.686623653609619*pi,0.28212265261509994*pi) q[26];
U1q(0.571756430086835*pi,1.6011416315925793*pi) q[27];
U1q(0.532440618472141*pi,1.4703611104017504*pi) q[28];
U1q(0.462401371277584*pi,1.7296317900798108*pi) q[29];
U1q(0.120730825742941*pi,0.49082493165682983*pi) q[30];
U1q(0.652752956125529*pi,1.8389097318274201*pi) q[31];
U1q(0.4081851146956*pi,0.01680445740115033*pi) q[32];
U1q(0.764183565601364*pi,0.5786762718387299*pi) q[33];
U1q(0.246262839029821*pi,1.8250655750920597*pi) q[34];
U1q(0.0836523293206387*pi,1.4277936064556194*pi) q[35];
U1q(0.499569754315778*pi,1.65449980077031*pi) q[36];
U1q(0.81047344170317*pi,0.9431947623206298*pi) q[37];
U1q(0.346701859866767*pi,1.8770515968423407*pi) q[38];
U1q(0.795880064048414*pi,1.84848476393788*pi) q[39];
RZZ(0.5*pi) q[38],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[4],q[33];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[6],q[14];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[9],q[8];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[12],q[22];
RZZ(0.5*pi) q[36],q[13];
RZZ(0.5*pi) q[16],q[35];
RZZ(0.5*pi) q[17],q[20];
RZZ(0.5*pi) q[18],q[19];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[29],q[30];
RZZ(0.5*pi) q[37],q[39];
U1q(0.639169713842014*pi,1.21804514922718*pi) q[0];
U1q(0.385562928529215*pi,1.1946088613933004*pi) q[1];
U1q(0.887863149989967*pi,1.6212886971888008*pi) q[2];
U1q(0.167044164766264*pi,0.7257847536154003*pi) q[3];
U1q(0.560223936729999*pi,1.9180737428402992*pi) q[4];
U1q(0.341566938830231*pi,0.42139288742989933*pi) q[5];
U1q(0.720317640504433*pi,0.5971699498233392*pi) q[6];
U1q(0.472763632795496*pi,1.5273371784645402*pi) q[7];
U1q(0.620864008057886*pi,1.5624119373900296*pi) q[8];
U1q(0.831829128247122*pi,1.7395890769827993*pi) q[9];
U1q(0.330106354344006*pi,1.6407658681974002*pi) q[10];
U1q(0.462558046338378*pi,1.2571264502040993*pi) q[11];
U1q(0.285263490473877*pi,0.1639231066899498*pi) q[12];
U1q(0.502312231071644*pi,1.1456260631151007*pi) q[13];
U1q(0.666925627481641*pi,0.1434533042800501*pi) q[14];
U1q(0.775481794957139*pi,0.578055618054*pi) q[15];
U1q(0.847433145086048*pi,1.8184194804782896*pi) q[16];
U1q(0.950124540989318*pi,0.3455931726813297*pi) q[17];
U1q(0.752239596582223*pi,1.6361187475265506*pi) q[18];
U1q(0.450982790194964*pi,1.2694217207397998*pi) q[19];
U1q(0.283172339966244*pi,1.1197075692602096*pi) q[20];
U1q(0.546011362998263*pi,0.34203437048527086*pi) q[21];
U1q(0.31960392750246*pi,0.8135315372190499*pi) q[22];
U1q(0.614175927919093*pi,1.0044997353585803*pi) q[23];
U1q(0.432171556379879*pi,1.5624623888048195*pi) q[24];
U1q(0.296837443083669*pi,1.3524224845052206*pi) q[25];
U1q(0.840233686872041*pi,0.4129663812178004*pi) q[26];
U1q(0.318598078909982*pi,1.8078382754739*pi) q[27];
U1q(0.367624761377996*pi,0.48252927103060017*pi) q[28];
U1q(0.155641086619243*pi,0.8575214989160003*pi) q[29];
U1q(0.80392440035047*pi,1.9988496661584492*pi) q[30];
U1q(0.347500558652029*pi,0.6084257384667406*pi) q[31];
U1q(0.289774377938883*pi,0.7315435047965604*pi) q[32];
U1q(0.434439084420158*pi,1.2059905516559493*pi) q[33];
U1q(0.777166625262875*pi,1.8535892859485994*pi) q[34];
U1q(0.138348466345774*pi,1.6353332718823008*pi) q[35];
U1q(0.458696502499379*pi,0.06847235165461019*pi) q[36];
U1q(0.590342794481657*pi,1.5172447693628008*pi) q[37];
U1q(0.240873278193524*pi,1.5726906007774009*pi) q[38];
U1q(0.49395971904791*pi,0.6919375299220292*pi) q[39];
RZZ(0.5*pi) q[6],q[0];
RZZ(0.5*pi) q[1],q[15];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[3],q[10];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[8],q[18];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[30],q[12];
RZZ(0.5*pi) q[39],q[13];
RZZ(0.5*pi) q[21],q[14];
RZZ(0.5*pi) q[16],q[19];
RZZ(0.5*pi) q[20],q[34];
RZZ(0.5*pi) q[29],q[22];
RZZ(0.5*pi) q[23],q[37];
RZZ(0.5*pi) q[24],q[27];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[26],q[33];
RZZ(0.5*pi) q[28],q[35];
RZZ(0.5*pi) q[38],q[36];
U1q(0.485046167767501*pi,1.8989099840021009*pi) q[0];
U1q(0.810125685635679*pi,1.2681180463234991*pi) q[1];
U1q(0.853349130191832*pi,1.6695133794915993*pi) q[2];
U1q(0.225485772733629*pi,1.8794742388542893*pi) q[3];
U1q(0.404997849558503*pi,0.08190834862229934*pi) q[4];
U1q(0.476645858186974*pi,0.9611997827273004*pi) q[5];
U1q(0.30197748804202*pi,1.7256925624702006*pi) q[6];
U1q(0.520565324187245*pi,1.2929010263526006*pi) q[7];
U1q(0.723075585947422*pi,0.7153866327873306*pi) q[8];
U1q(0.489761493779094*pi,0.2972976899217006*pi) q[9];
U1q(0.533849801602835*pi,1.0782414529194*pi) q[10];
U1q(0.717012326890099*pi,1.6091651110387009*pi) q[11];
U1q(0.361798396146727*pi,0.2842914046839091*pi) q[12];
U1q(0.231474840388114*pi,0.3379125290977001*pi) q[13];
U1q(0.733221126495442*pi,0.8372874599726003*pi) q[14];
U1q(0.720232079774362*pi,1.8913689309597004*pi) q[15];
U1q(0.516570671454139*pi,1.7559101971787996*pi) q[16];
U1q(0.391119863776588*pi,0.4887734364285006*pi) q[17];
U1q(0.334976341999809*pi,0.6547533546373003*pi) q[18];
U1q(0.726603998217709*pi,1.1401434072840004*pi) q[19];
U1q(0.357348482841429*pi,1.35407594400594*pi) q[20];
U1q(0.339167311386811*pi,1.4980621147762996*pi) q[21];
U1q(0.377664335652551*pi,0.6504835171025007*pi) q[22];
U1q(0.49516495185273*pi,0.17089067178119954*pi) q[23];
U1q(0.301778477860536*pi,0.6217472500882995*pi) q[24];
U1q(0.356441529407207*pi,0.4918916913981004*pi) q[25];
U1q(0.247873582104482*pi,0.08724318699908995*pi) q[26];
U1q(0.788302774144042*pi,0.36605421934110005*pi) q[27];
U1q(0.544814761278199*pi,0.7538252866070003*pi) q[28];
U1q(0.515161108155073*pi,0.8220338305513994*pi) q[29];
U1q(0.459400506638106*pi,0.7623404035267995*pi) q[30];
U1q(0.672251836782594*pi,1.0552252205639991*pi) q[31];
U1q(0.353984502210819*pi,0.5269709944033991*pi) q[32];
U1q(0.538169014833147*pi,0.6406732652085996*pi) q[33];
U1q(0.317293157315052*pi,1.7165668013646993*pi) q[34];
U1q(0.281310281354824*pi,1.3904562623950003*pi) q[35];
U1q(0.760808569500946*pi,0.8036708988339996*pi) q[36];
U1q(0.388692133424015*pi,0.6398101408433998*pi) q[37];
U1q(0.886503514790454*pi,0.1592645737054994*pi) q[38];
U1q(0.589259604196514*pi,1.1054295310452993*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[1],q[32];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[3],q[27];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[7],q[11];
RZZ(0.5*pi) q[8],q[39];
RZZ(0.5*pi) q[9],q[13];
RZZ(0.5*pi) q[28],q[10];
RZZ(0.5*pi) q[12],q[35];
RZZ(0.5*pi) q[15],q[14];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[23],q[18];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[25],q[21];
RZZ(0.5*pi) q[36],q[22];
RZZ(0.5*pi) q[30],q[24];
RZZ(0.5*pi) q[31],q[26];
RZZ(0.5*pi) q[37],q[33];
U1q(0.272689373423054*pi,0.07646934387179982*pi) q[0];
U1q(0.740926587178512*pi,0.7496002871561984*pi) q[1];
U1q(0.660727620726012*pi,0.8760910430239992*pi) q[2];
U1q(0.205642532518183*pi,1.5569877245025996*pi) q[3];
U1q(0.282962628458005*pi,1.8673098773878003*pi) q[4];
U1q(0.748248624946222*pi,0.008798308595299176*pi) q[5];
U1q(0.535119018128953*pi,1.6000870027498006*pi) q[6];
U1q(0.74893708858858*pi,1.0199042575099*pi) q[7];
U1q(0.40138225870397*pi,0.5661672772040998*pi) q[8];
U1q(0.252480948316796*pi,1.8891669171341015*pi) q[9];
U1q(0.298835414078385*pi,0.3516453203660994*pi) q[10];
U1q(0.842885422595933*pi,1.6578363961414997*pi) q[11];
U1q(0.395511638196745*pi,0.7079841007032996*pi) q[12];
U1q(0.166317013475876*pi,0.7943738524088992*pi) q[13];
U1q(0.377618613423848*pi,0.7814253741477*pi) q[14];
U1q(0.515119813558816*pi,0.5602377351804009*pi) q[15];
U1q(0.684406229694427*pi,1.2165795997551*pi) q[16];
U1q(0.519826657939295*pi,1.937380294097899*pi) q[17];
U1q(0.600895154367061*pi,0.7010050298300001*pi) q[18];
U1q(0.586756369048285*pi,0.3893345780142994*pi) q[19];
U1q(0.637651795904757*pi,0.7623503389347999*pi) q[20];
U1q(0.85072561463764*pi,1.1138603819512003*pi) q[21];
U1q(0.77373353313932*pi,0.9779194289463007*pi) q[22];
U1q(0.257382370876835*pi,0.28299203414080054*pi) q[23];
U1q(0.375873386233571*pi,0.23962082369670057*pi) q[24];
U1q(0.69593834459832*pi,0.5033078241279991*pi) q[25];
U1q(0.280571178846466*pi,0.6151845056739607*pi) q[26];
U1q(0.35012543199078*pi,1.1136859266611996*pi) q[27];
U1q(0.646952441179745*pi,1.7829142736002996*pi) q[28];
U1q(0.0600090520845183*pi,0.32295671806529924*pi) q[29];
U1q(0.6559958583797*pi,0.34918599893100044*pi) q[30];
U1q(0.174682433667661*pi,1.8836949062879995*pi) q[31];
U1q(0.405431583533657*pi,1.3775931476906003*pi) q[32];
U1q(0.348023046867191*pi,1.3698844877373002*pi) q[33];
U1q(0.587780326045055*pi,1.6279725888719998*pi) q[34];
U1q(0.454521168319484*pi,0.8452391538423996*pi) q[35];
U1q(0.352695475919976*pi,0.07502891038370052*pi) q[36];
U1q(0.746639046558849*pi,0.38640838684679935*pi) q[37];
U1q(0.32718452576656*pi,0.7449933280609002*pi) q[38];
U1q(0.529480161558753*pi,1.5951122069627992*pi) q[39];
RZZ(0.5*pi) q[26],q[0];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[38],q[2];
RZZ(0.5*pi) q[21],q[3];
RZZ(0.5*pi) q[4],q[28];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[20],q[6];
RZZ(0.5*pi) q[7],q[33];
RZZ(0.5*pi) q[8],q[10];
RZZ(0.5*pi) q[29],q[9];
RZZ(0.5*pi) q[11],q[13];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[37],q[14];
RZZ(0.5*pi) q[17],q[15];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[39],q[24];
U1q(0.333093946277537*pi,0.7949504302460006*pi) q[0];
U1q(0.444148385369062*pi,1.6075348794744002*pi) q[1];
U1q(0.331158323141311*pi,0.5276635041639999*pi) q[2];
U1q(0.464804537936317*pi,1.4485363833662994*pi) q[3];
U1q(0.347231192150154*pi,1.8280182461049002*pi) q[4];
U1q(0.542626720609545*pi,1.8877602020757998*pi) q[5];
U1q(0.138696923056326*pi,0.19522813766350033*pi) q[6];
U1q(0.621450038933918*pi,1.0218666012754998*pi) q[7];
U1q(0.802225204031235*pi,0.01358908380879953*pi) q[8];
U1q(0.517309016424964*pi,0.6975543942713003*pi) q[9];
U1q(0.187442143194181*pi,1.092995395210199*pi) q[10];
U1q(0.382039742165076*pi,1.1421382876874002*pi) q[11];
U1q(0.662442340709105*pi,0.23281441172290052*pi) q[12];
U1q(0.269606294682352*pi,1.242083990071901*pi) q[13];
U1q(0.825394340266734*pi,0.30622213059779924*pi) q[14];
U1q(0.907108156049125*pi,0.569436726072901*pi) q[15];
U1q(0.892937807126818*pi,1.8998377010845005*pi) q[16];
U1q(0.524064372074674*pi,0.7332196145251011*pi) q[17];
U1q(0.784035861716478*pi,1.2608335056763007*pi) q[18];
U1q(0.395544270735603*pi,1.6512429325422993*pi) q[19];
U1q(0.319229298815783*pi,1.3333358184474005*pi) q[20];
U1q(0.686369971256301*pi,0.5292963977074017*pi) q[21];
U1q(0.710850844543759*pi,1.4716742424620008*pi) q[22];
U1q(0.514495284624502*pi,0.3780723985639014*pi) q[23];
U1q(0.335790577922624*pi,0.1501665666499008*pi) q[24];
U1q(0.817660529010691*pi,0.45360753701529966*pi) q[25];
U1q(0.592278987837679*pi,1.2694043530350996*pi) q[26];
U1q(0.676206209985493*pi,0.5603630025312007*pi) q[27];
U1q(0.612225874625805*pi,0.9252064863302998*pi) q[28];
U1q(0.703225546117443*pi,1.834682692759099*pi) q[29];
U1q(0.386756287198544*pi,0.05075882821440025*pi) q[30];
U1q(0.25603667657484*pi,0.058273799039600505*pi) q[31];
U1q(0.165491411255117*pi,0.13775184485819914*pi) q[32];
U1q(0.677010351077071*pi,0.37462296506009984*pi) q[33];
U1q(0.487105365281788*pi,1.1530858827277992*pi) q[34];
U1q(0.114390431778676*pi,0.30886588836549933*pi) q[35];
U1q(0.289822028301438*pi,0.6434882596139992*pi) q[36];
U1q(0.753628537207985*pi,0.053792147492199405*pi) q[37];
U1q(0.680287835069673*pi,1.7412396711489002*pi) q[38];
U1q(0.657770284737689*pi,0.6877438794895987*pi) q[39];
RZZ(0.5*pi) q[27],q[0];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[29],q[2];
RZZ(0.5*pi) q[31],q[3];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[5],q[22];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[8],q[12];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[32],q[10];
RZZ(0.5*pi) q[16],q[13];
RZZ(0.5*pi) q[38],q[14];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[28],q[24];
RZZ(0.5*pi) q[25],q[33];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[34],q[37];
U1q(0.360541353297399*pi,1.0994669666922015*pi) q[0];
U1q(0.525476235344274*pi,0.28028077379520155*pi) q[1];
U1q(0.926836698740279*pi,1.3373547613862016*pi) q[2];
U1q(0.249086168943163*pi,1.4070103616432998*pi) q[3];
U1q(0.422430290352584*pi,1.7118660740085012*pi) q[4];
U1q(0.878274325499196*pi,1.4287405436223999*pi) q[5];
U1q(0.788724869950132*pi,1.2750601421325989*pi) q[6];
U1q(0.713108164278136*pi,0.33645950283560033*pi) q[7];
U1q(0.44708829456635*pi,0.3429318798647998*pi) q[8];
U1q(0.0901354131491885*pi,1.8106036297381003*pi) q[9];
U1q(0.68859620043723*pi,1.9225992785299013*pi) q[10];
U1q(0.750389303365724*pi,0.8388222173103017*pi) q[11];
U1q(0.514056955238715*pi,1.4824078530542018*pi) q[12];
U1q(0.855514312167377*pi,1.7162999140474007*pi) q[13];
U1q(0.420259434547998*pi,1.1165808809851008*pi) q[14];
U1q(0.282928088481884*pi,0.43963767688549993*pi) q[15];
U1q(0.462260222005264*pi,1.0692196548653001*pi) q[16];
U1q(0.565803677547889*pi,0.03315946699559902*pi) q[17];
U1q(0.411112924259974*pi,1.7824865392768992*pi) q[18];
U1q(0.614004780469149*pi,1.9453776138451992*pi) q[19];
U1q(0.372054715781059*pi,0.6050573492713003*pi) q[20];
U1q(0.606081043346575*pi,0.2590353602151012*pi) q[21];
U1q(0.568916538805986*pi,1.5734931120259006*pi) q[22];
U1q(0.36100603956462*pi,1.2644224205408001*pi) q[23];
U1q(0.241937423791678*pi,1.6173713245004997*pi) q[24];
U1q(0.371199006933256*pi,1.4278193903075014*pi) q[25];
U1q(0.719489953055306*pi,1.3928257572323002*pi) q[26];
U1q(0.203525204101118*pi,1.525317374462599*pi) q[27];
U1q(0.743849407046656*pi,1.3677864855902016*pi) q[28];
U1q(0.469638888405315*pi,1.3557421840162007*pi) q[29];
U1q(0.462743363618117*pi,0.2559125872373009*pi) q[30];
U1q(0.212272552866127*pi,1.4801589388594003*pi) q[31];
U1q(0.56652387196782*pi,1.9769448552175994*pi) q[32];
U1q(0.604166923064851*pi,1.534277176950301*pi) q[33];
U1q(0.121241439395294*pi,0.6363372467091999*pi) q[34];
U1q(0.581195990472228*pi,0.34722896596860053*pi) q[35];
U1q(0.600983904142197*pi,1.5452169901358985*pi) q[36];
U1q(0.39074877746691*pi,1.2266169456549*pi) q[37];
U1q(0.272457577187803*pi,0.13580738043309992*pi) q[38];
U1q(0.0746941621233767*pi,1.8954922883568983*pi) q[39];
RZZ(0.5*pi) q[15],q[0];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[2],q[24];
RZZ(0.5*pi) q[3],q[30];
RZZ(0.5*pi) q[4],q[20];
RZZ(0.5*pi) q[14],q[5];
RZZ(0.5*pi) q[31],q[6];
RZZ(0.5*pi) q[7],q[29];
RZZ(0.5*pi) q[8],q[21];
RZZ(0.5*pi) q[9],q[37];
RZZ(0.5*pi) q[33],q[10];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[16],q[39];
RZZ(0.5*pi) q[17],q[35];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[19],q[27];
RZZ(0.5*pi) q[38],q[22];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[32],q[28];
U1q(0.14152372749624*pi,0.031096785912499314*pi) q[0];
U1q(0.0322411078204634*pi,0.6166082027775985*pi) q[1];
U1q(0.447770704201286*pi,1.2241654109045008*pi) q[2];
U1q(0.340806717083723*pi,0.3820589794907008*pi) q[3];
U1q(0.353908259641203*pi,1.095530601603599*pi) q[4];
U1q(0.597278813849119*pi,1.1718617085153014*pi) q[5];
U1q(0.449173875646482*pi,1.0929309587772984*pi) q[6];
U1q(0.540149959976407*pi,1.3055203543041003*pi) q[7];
U1q(0.564765433764938*pi,0.2402130361755006*pi) q[8];
U1q(0.355360857388817*pi,1.0709616364075991*pi) q[9];
U1q(0.408642882972995*pi,0.34530463300169956*pi) q[10];
U1q(0.607380111179703*pi,1.8077348350410993*pi) q[11];
U1q(0.766290876503306*pi,0.4186513368639986*pi) q[12];
U1q(0.731950865209254*pi,1.2823391015454*pi) q[13];
U1q(0.0428848701473328*pi,1.1682956362582004*pi) q[14];
U1q(0.615893632879497*pi,1.3019031482591998*pi) q[15];
U1q(0.471442316200409*pi,1.1743134009524994*pi) q[16];
U1q(0.928030682889834*pi,1.0836864818361*pi) q[17];
U1q(0.551793603279744*pi,0.8106863642178013*pi) q[18];
U1q(0.418115081174563*pi,1.7765958679376013*pi) q[19];
U1q(0.725965579485019*pi,1.706330604064501*pi) q[20];
U1q(0.855681414615226*pi,1.6135407706115998*pi) q[21];
U1q(0.540456277312879*pi,0.04896704531489959*pi) q[22];
U1q(0.799255411416524*pi,1.4968330474913003*pi) q[23];
U1q(0.458365273242125*pi,1.1560921678551992*pi) q[24];
U1q(0.391941042588094*pi,1.8008104314213007*pi) q[25];
U1q(0.46203795928401*pi,1.6502239942599992*pi) q[26];
U1q(0.826810515962961*pi,1.0903837229854005*pi) q[27];
U1q(0.681942202878676*pi,0.5180729530704014*pi) q[28];
U1q(0.654944312076029*pi,1.665311505873401*pi) q[29];
U1q(0.460760222041635*pi,0.6856783720268993*pi) q[30];
U1q(0.818654713653356*pi,0.37400396415380044*pi) q[31];
U1q(0.320893634135305*pi,1.1580172205058012*pi) q[32];
U1q(0.458417010315943*pi,1.8286883708112995*pi) q[33];
U1q(0.85829298810288*pi,0.6818320652624017*pi) q[34];
U1q(0.138001447430043*pi,0.567417501781101*pi) q[35];
U1q(0.228079205985453*pi,1.374284903867899*pi) q[36];
U1q(0.227241783815173*pi,0.14039715301059985*pi) q[37];
U1q(0.262815300736987*pi,0.9022078444906008*pi) q[38];
U1q(0.516240288328946*pi,1.2590163033714*pi) q[39];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[1],q[30];
RZZ(0.5*pi) q[21],q[2];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[25],q[4];
RZZ(0.5*pi) q[24],q[5];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[32],q[9];
RZZ(0.5*pi) q[37],q[10];
RZZ(0.5*pi) q[11],q[19];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[15],q[22];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[31],q[18];
RZZ(0.5*pi) q[26],q[36];
RZZ(0.5*pi) q[39],q[27];
RZZ(0.5*pi) q[29],q[34];
RZZ(0.5*pi) q[38],q[35];
U1q(0.125835454006992*pi,0.042092177112500906*pi) q[0];
U1q(0.732212148161911*pi,0.4849983156010005*pi) q[1];
U1q(0.260557482831644*pi,0.5903148170812997*pi) q[2];
U1q(0.473562986592926*pi,1.5419445212009997*pi) q[3];
U1q(0.420961605684617*pi,0.21620019323739825*pi) q[4];
U1q(0.564744178990841*pi,1.135706309610299*pi) q[5];
U1q(0.198959361641117*pi,0.1718015215198001*pi) q[6];
U1q(0.869751846770903*pi,0.37175034433670007*pi) q[7];
U1q(0.50453798761614*pi,0.3672008866017009*pi) q[8];
U1q(0.743205242028707*pi,1.9534222795006997*pi) q[9];
U1q(0.563087240911915*pi,0.730200266967401*pi) q[10];
U1q(0.416535808182887*pi,0.3096822919345996*pi) q[11];
U1q(0.518928393502844*pi,1.3994868007758008*pi) q[12];
U1q(0.703116902829537*pi,1.0777780607561986*pi) q[13];
U1q(0.551926215872259*pi,0.1384986161470998*pi) q[14];
U1q(0.397734548070865*pi,1.8754134434506007*pi) q[15];
U1q(0.699158720374696*pi,0.2872939625961983*pi) q[16];
U1q(0.185507852178553*pi,1.448342343613799*pi) q[17];
U1q(0.338988483849101*pi,0.48817053572189906*pi) q[18];
U1q(0.614490297243578*pi,0.1355791220284992*pi) q[19];
U1q(0.258248781732303*pi,1.9715069244042986*pi) q[20];
U1q(0.172160665340282*pi,1.5233963784045983*pi) q[21];
U1q(0.815340362381586*pi,0.7764443370721992*pi) q[22];
U1q(0.581348071691551*pi,0.29128638815700114*pi) q[23];
U1q(0.700624495949713*pi,0.7166475812433006*pi) q[24];
U1q(0.793390750938045*pi,1.2590958040053017*pi) q[25];
U1q(0.51032025253361*pi,1.9189400268461014*pi) q[26];
U1q(0.59020143755777*pi,0.7337781937847012*pi) q[27];
U1q(0.383662232495717*pi,0.1005161687156999*pi) q[28];
U1q(0.518233017140261*pi,1.6311828421961003*pi) q[29];
U1q(0.437114687047119*pi,0.8044319014653993*pi) q[30];
U1q(0.70494327898113*pi,0.649394664794201*pi) q[31];
U1q(0.504353643527446*pi,1.607143456651201*pi) q[32];
U1q(0.42301566433801*pi,1.688001662882499*pi) q[33];
U1q(0.64875222019692*pi,1.0100308347804017*pi) q[34];
U1q(0.92035529300504*pi,0.7778612356568004*pi) q[35];
U1q(0.438069120815617*pi,1.4275208596956013*pi) q[36];
U1q(0.172056524520363*pi,0.22066362329509914*pi) q[37];
U1q(0.324780271149357*pi,1.3177306748852011*pi) q[38];
U1q(0.660962299885425*pi,1.1090715806600997*pi) q[39];
rz(1.8415721487812995*pi) q[0];
rz(1.0750896977594984*pi) q[1];
rz(1.7670152980233986*pi) q[2];
rz(0.07587378131820088*pi) q[3];
rz(1.346668980568701*pi) q[4];
rz(3.129699040873799*pi) q[5];
rz(3.5455418017903*pi) q[6];
rz(1.6280401872888*pi) q[7];
rz(0.506397120892899*pi) q[8];
rz(3.1938758266843017*pi) q[9];
rz(2.261726586901599*pi) q[10];
rz(0.6751021164369*pi) q[11];
rz(2.2802690307924998*pi) q[12];
rz(3.709948231565999*pi) q[13];
rz(0.6236735615738986*pi) q[14];
rz(2.6721306180523996*pi) q[15];
rz(3.7064299726804*pi) q[16];
rz(1.4344146749009*pi) q[17];
rz(2.0839944102174*pi) q[18];
rz(3.4872995968725*pi) q[19];
rz(2.1448398540252*pi) q[20];
rz(2.2289941331886*pi) q[21];
rz(2.5739984483805003*pi) q[22];
rz(2.5072652281487997*pi) q[23];
rz(0.09088260946699833*pi) q[24];
rz(1.1841307746386*pi) q[25];
rz(0.04101902204070029*pi) q[26];
rz(1.3749059067581015*pi) q[27];
rz(0.06487776011089963*pi) q[28];
rz(0.09358203526059938*pi) q[29];
rz(1.6546786931255006*pi) q[30];
rz(0.641091278865801*pi) q[31];
rz(1.2476804228431995*pi) q[32];
rz(3.9486469632925996*pi) q[33];
rz(1.765168820378701*pi) q[34];
rz(0.6824608431880002*pi) q[35];
rz(2.4865700993320985*pi) q[36];
rz(2.411592784573301*pi) q[37];
rz(3.7394178548423014*pi) q[38];
rz(1.8279220294691996*pi) q[39];
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