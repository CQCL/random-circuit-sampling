OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
rz(2.61392386214776*pi) q[0];
rz(0.6819457969735632*pi) q[1];
rz(3.566545370905034*pi) q[2];
rz(3.642101029496402*pi) q[3];
rz(1.7205930677294219*pi) q[4];
rz(3.809257852791584*pi) q[5];
rz(3.574253015234487*pi) q[6];
rz(3.9456434539791982*pi) q[7];
rz(1.93281879249718*pi) q[8];
rz(3.668333222820514*pi) q[9];
rz(1.85071493531295*pi) q[10];
rz(1.7002046584409631*pi) q[11];
rz(1.86285544842724*pi) q[12];
rz(3.404111978249527*pi) q[13];
rz(0.686170821743973*pi) q[14];
rz(3.193163438164065*pi) q[15];
rz(3.602621789736238*pi) q[16];
rz(0.468944406180575*pi) q[17];
rz(3.600903770175533*pi) q[18];
rz(2.1462992106763403*pi) q[19];
rz(2.529620922644966*pi) q[20];
rz(1.15176152834191*pi) q[21];
rz(3.111624991406239*pi) q[22];
rz(0.04206693093207731*pi) q[23];
rz(3.388818572301492*pi) q[24];
rz(3.427077394209366*pi) q[25];
rz(1.11700015701351*pi) q[26];
rz(3.0693517092778038*pi) q[27];
rz(3.634790817958445*pi) q[28];
rz(0.705887861709461*pi) q[29];
rz(2.23182130892443*pi) q[30];
rz(3.5722594552160367*pi) q[31];
rz(1.73987139788451*pi) q[32];
rz(1.3284463138031797*pi) q[33];
rz(1.1771591536627586*pi) q[34];
rz(0.2590360322108035*pi) q[35];
rz(1.43902920708924*pi) q[36];
rz(0.44694347488295116*pi) q[37];
rz(1.0691570572804758*pi) q[38];
rz(2.36609293577412*pi) q[39];
U1q(0.693343935619707*pi,1.5474096713285301*pi) q[0];
U1q(3.335324176261267*pi,1.80269161589785*pi) q[1];
U1q(0.602503207631316*pi,0.223493058369388*pi) q[2];
U1q(1.62149145963868*pi,0.501346525830488*pi) q[3];
U1q(1.32932035962142*pi,0.877577696870966*pi) q[4];
U1q(0.644809326233638*pi,0.181467463516195*pi) q[5];
U1q(0.767585350249221*pi,0.517885092385709*pi) q[6];
U1q(1.33603747339688*pi,0.461648646861168*pi) q[7];
U1q(0.697346547428285*pi,1.4596625517524*pi) q[8];
U1q(0.403031258579952*pi,1.018889258512612*pi) q[9];
U1q(0.741044050570865*pi,1.12182549852924*pi) q[10];
U1q(1.34228730746813*pi,1.38085959259999*pi) q[11];
U1q(0.860802446046296*pi,1.0334586456319*pi) q[12];
U1q(1.4937367753478*pi,1.787408552074843*pi) q[13];
U1q(0.403089429986743*pi,0.42809605108606*pi) q[14];
U1q(1.12939807020756*pi,0.00730561315468248*pi) q[15];
U1q(0.340424899184175*pi,1.587326532142358*pi) q[16];
U1q(1.68903778227657*pi,1.9526234009633918*pi) q[17];
U1q(1.73010927096583*pi,1.214307634387986*pi) q[18];
U1q(0.696920630896001*pi,1.683019093777735*pi) q[19];
U1q(1.49326441192771*pi,1.176043460054607*pi) q[20];
U1q(0.251211307774309*pi,0.820599638252661*pi) q[21];
U1q(3.67403052936587*pi,1.9601349423265453*pi) q[22];
U1q(1.59873736039172*pi,0.539812810850209*pi) q[23];
U1q(0.886443345703264*pi,1.758458844102353*pi) q[24];
U1q(0.764983072679902*pi,0.171554168845304*pi) q[25];
U1q(1.45058831492519*pi,0.470672303547566*pi) q[26];
U1q(0.78949093939519*pi,1.698204039903121*pi) q[27];
U1q(0.496172055243047*pi,1.848129814004309*pi) q[28];
U1q(0.19094813053126*pi,0.960236767478401*pi) q[29];
U1q(0.594899108722984*pi,1.4765737072531269*pi) q[30];
U1q(1.70416287013194*pi,1.303039562340028*pi) q[31];
U1q(0.639812432906418*pi,0.974213580985569*pi) q[32];
U1q(1.83780783689849*pi,0.506641042184019*pi) q[33];
U1q(1.5249543026962*pi,0.233970930678786*pi) q[34];
U1q(1.85480741809977*pi,0.381110833164711*pi) q[35];
U1q(0.711538258296304*pi,0.543311254607767*pi) q[36];
U1q(1.85426719522149*pi,1.491512339592152*pi) q[37];
U1q(1.81407605729664*pi,0.324439393226768*pi) q[38];
U1q(0.821305106142815*pi,1.1778286521432522*pi) q[39];
RZZ(0.0*pi) q[6],q[0];
RZZ(0.0*pi) q[1],q[25];
RZZ(0.0*pi) q[23],q[2];
RZZ(0.0*pi) q[3],q[8];
RZZ(0.0*pi) q[4],q[9];
RZZ(0.0*pi) q[5],q[19];
RZZ(0.0*pi) q[7],q[17];
RZZ(0.0*pi) q[37],q[10];
RZZ(0.0*pi) q[11],q[33];
RZZ(0.0*pi) q[12],q[28];
RZZ(0.0*pi) q[13],q[14];
RZZ(0.0*pi) q[15],q[29];
RZZ(0.0*pi) q[27],q[16];
RZZ(0.0*pi) q[38],q[18];
RZZ(0.0*pi) q[20],q[32];
RZZ(0.0*pi) q[26],q[21];
RZZ(0.0*pi) q[31],q[22];
RZZ(0.0*pi) q[24],q[35];
RZZ(0.0*pi) q[30],q[34];
RZZ(0.0*pi) q[39],q[36];
rz(3.899704490477267*pi) q[0];
rz(0.747953692648218*pi) q[1];
rz(3.612844241346123*pi) q[2];
rz(2.45929917459336*pi) q[3];
rz(3.00615150575844*pi) q[4];
rz(1.34569225338001*pi) q[5];
rz(3.566421065720003*pi) q[6];
rz(3.201298236930616*pi) q[7];
rz(0.343134154246935*pi) q[8];
rz(3.168007618995339*pi) q[9];
rz(0.7426386127341*pi) q[10];
rz(1.27872537045198*pi) q[11];
rz(0.30883493665941*pi) q[12];
rz(2.5645907858465202*pi) q[13];
rz(2.4071675642426*pi) q[14];
rz(2.49763923142854*pi) q[15];
rz(3.736351312442532*pi) q[16];
rz(0.360332361362738*pi) q[17];
rz(0.540225508689411*pi) q[18];
rz(0.433852208977704*pi) q[19];
rz(0.626929336457806*pi) q[20];
rz(0.990007612206155*pi) q[21];
rz(3.476461370024508*pi) q[22];
rz(3.620415402070862*pi) q[23];
rz(2.4654409424648698*pi) q[24];
rz(2.55353843160304*pi) q[25];
rz(3.13834920957146*pi) q[26];
rz(0.924562535738567*pi) q[27];
rz(0.677965957451838*pi) q[28];
rz(1.26591639535764*pi) q[29];
rz(1.40335283242959*pi) q[30];
rz(0.979337856510286*pi) q[31];
rz(0.426391951291058*pi) q[32];
rz(2.03538269698272*pi) q[33];
rz(0.448238322498666*pi) q[34];
rz(0.481038563631261*pi) q[35];
rz(0.994078077445272*pi) q[36];
rz(1.42965260597514*pi) q[37];
rz(2.43789912477737*pi) q[38];
rz(1.27072178839186*pi) q[39];
U1q(0.112672794539398*pi,1.848908962092741*pi) q[0];
U1q(0.358827672798324*pi,1.13093989165865*pi) q[1];
U1q(0.19855954989146*pi,0.167718343322642*pi) q[2];
U1q(0.761522560623669*pi,1.293737494073643*pi) q[3];
U1q(0.899453461027655*pi,0.110177455594578*pi) q[4];
U1q(0.34391530122347*pi,0.477168131336792*pi) q[5];
U1q(0.411765245646519*pi,1.9487540132279966*pi) q[6];
U1q(0.68737579848881*pi,1.769192706208781*pi) q[7];
U1q(0.634479329981367*pi,0.161262792174941*pi) q[8];
U1q(0.656225783274493*pi,1.902913046255951*pi) q[9];
U1q(0.797314203606484*pi,0.20863834634219*pi) q[10];
U1q(0.450739552357126*pi,0.755991712353891*pi) q[11];
U1q(0.651384013176014*pi,0.0359325582270763*pi) q[12];
U1q(0.850236020986048*pi,1.625720710676418*pi) q[13];
U1q(0.588289820419951*pi,1.495659498629807*pi) q[14];
U1q(0.826500696596189*pi,1.9685460264027617*pi) q[15];
U1q(0.354547264038934*pi,1.712457455312858*pi) q[16];
U1q(0.76405281182309*pi,0.669372156812962*pi) q[17];
U1q(0.158496838589417*pi,1.03523031409519*pi) q[18];
U1q(0.413300922626101*pi,1.267818415692658*pi) q[19];
U1q(0.091223427099478*pi,1.571218717658971*pi) q[20];
U1q(0.488679696879294*pi,1.28347603545039*pi) q[21];
U1q(0.650090223796699*pi,1.511487628368446*pi) q[22];
U1q(0.458016437504262*pi,0.515114466130138*pi) q[23];
U1q(0.743122814063918*pi,1.550994701657378*pi) q[24];
U1q(0.555339356150773*pi,1.9362037944110582*pi) q[25];
U1q(0.638744837552241*pi,1.842000837793843*pi) q[26];
U1q(0.471671618650692*pi,0.644758334615391*pi) q[27];
U1q(0.497964648904691*pi,0.276848677064561*pi) q[28];
U1q(0.186263464557282*pi,0.366342833020219*pi) q[29];
U1q(0.387787286701087*pi,0.145902261814634*pi) q[30];
U1q(0.529949721812562*pi,0.170166572493179*pi) q[31];
U1q(0.344156258719445*pi,1.849725155797941*pi) q[32];
U1q(0.816078924465085*pi,1.448310042796693*pi) q[33];
U1q(0.498027145467997*pi,0.369579999244198*pi) q[34];
U1q(0.652115858031254*pi,0.820534367342925*pi) q[35];
U1q(0.379905012746608*pi,0.684552589812941*pi) q[36];
U1q(0.217492270061332*pi,0.726964319193091*pi) q[37];
U1q(0.642488569854318*pi,1.613150894674856*pi) q[38];
U1q(0.378086859555288*pi,1.10469754427413*pi) q[39];
RZZ(0.0*pi) q[4],q[0];
RZZ(0.0*pi) q[1],q[24];
RZZ(0.0*pi) q[20],q[2];
RZZ(0.0*pi) q[3],q[10];
RZZ(0.0*pi) q[5],q[16];
RZZ(0.0*pi) q[6],q[14];
RZZ(0.0*pi) q[7],q[38];
RZZ(0.0*pi) q[8],q[36];
RZZ(0.0*pi) q[9],q[11];
RZZ(0.0*pi) q[12],q[27];
RZZ(0.0*pi) q[13],q[21];
RZZ(0.0*pi) q[15],q[25];
RZZ(0.0*pi) q[17],q[19];
RZZ(0.0*pi) q[22],q[18];
RZZ(0.0*pi) q[32],q[23];
RZZ(0.0*pi) q[26],q[39];
RZZ(0.0*pi) q[34],q[28];
RZZ(0.0*pi) q[37],q[29];
RZZ(0.0*pi) q[30],q[31];
RZZ(0.0*pi) q[35],q[33];
rz(0.913153011501686*pi) q[0];
rz(0.584652987519556*pi) q[1];
rz(0.0681248849148746*pi) q[2];
rz(3.650277934721127*pi) q[3];
rz(3.777693948229968*pi) q[4];
rz(3.410990796994104*pi) q[5];
rz(0.974743894854002*pi) q[6];
rz(2.01290857817331*pi) q[7];
rz(2.7223986034880703*pi) q[8];
rz(1.48666650209425*pi) q[9];
rz(1.23272744263595*pi) q[10];
rz(2.68723813827325*pi) q[11];
rz(1.10628273090013*pi) q[12];
rz(0.440229799829984*pi) q[13];
rz(0.440225911015382*pi) q[14];
rz(2.81022601992886*pi) q[15];
rz(0.588426439910995*pi) q[16];
rz(0.211460020982574*pi) q[17];
rz(0.68717675728262*pi) q[18];
rz(0.385143985111408*pi) q[19];
rz(1.28147514662193*pi) q[20];
rz(3.5985994772063012*pi) q[21];
rz(3.93142839050582*pi) q[22];
rz(0.979672522394745*pi) q[23];
rz(1.14410056430435*pi) q[24];
rz(3.675038672453047*pi) q[25];
rz(0.820452355978208*pi) q[26];
rz(3.1691413647327042*pi) q[27];
rz(0.531977236637578*pi) q[28];
rz(0.0964123452633628*pi) q[29];
rz(3.98633225722937*pi) q[30];
rz(3.703367338265547*pi) q[31];
rz(3.348433041789408*pi) q[32];
rz(1.26988430223212*pi) q[33];
rz(0.914134496625363*pi) q[34];
rz(0.148185189900414*pi) q[35];
rz(2.86366126523743*pi) q[36];
rz(1.08845325766424*pi) q[37];
rz(1.43983189842237*pi) q[38];
rz(2.01380377217748*pi) q[39];
U1q(0.335929134546617*pi,1.819855930274272*pi) q[0];
U1q(0.762850896632044*pi,0.387621212789864*pi) q[1];
U1q(0.365505092569071*pi,1.05869053489883*pi) q[2];
U1q(0.445205368552782*pi,1.179493860748976*pi) q[3];
U1q(0.435057187617987*pi,0.538307982004721*pi) q[4];
U1q(0.98946886328525*pi,1.9367862964374953*pi) q[5];
U1q(0.279078406752584*pi,0.364314554158965*pi) q[6];
U1q(0.50551304676468*pi,1.459660747861423*pi) q[7];
U1q(0.832071454494486*pi,1.798142282984918*pi) q[8];
U1q(0.407759361564368*pi,1.860434798352465*pi) q[9];
U1q(0.847176649651305*pi,1.00552192673055*pi) q[10];
U1q(0.618414153940168*pi,1.379001567882348*pi) q[11];
U1q(0.114273545413377*pi,0.726287967982534*pi) q[12];
U1q(0.694813977594788*pi,0.741725784098332*pi) q[13];
U1q(0.783328320844114*pi,0.211651940014015*pi) q[14];
U1q(0.694493403025985*pi,1.4444048551132909*pi) q[15];
U1q(0.668427114458592*pi,0.586985613143863*pi) q[16];
U1q(0.415672344912941*pi,1.224070420879003*pi) q[17];
U1q(0.233369716692483*pi,0.080684299312991*pi) q[18];
U1q(0.430790885761373*pi,0.613400093128346*pi) q[19];
U1q(0.654531776337706*pi,1.03085714788177*pi) q[20];
U1q(0.391618890398795*pi,0.128049310803632*pi) q[21];
U1q(0.475978024720382*pi,1.1287133271456868*pi) q[22];
U1q(0.749729952272974*pi,0.391759953336476*pi) q[23];
U1q(0.916573042346116*pi,1.27718332652377*pi) q[24];
U1q(0.664681007813723*pi,0.139353799795608*pi) q[25];
U1q(0.735316861417305*pi,0.644282039495178*pi) q[26];
U1q(0.708480349614522*pi,1.8317842574061869*pi) q[27];
U1q(0.619596333804529*pi,0.451325308245538*pi) q[28];
U1q(0.514274645242525*pi,1.9933533593301371*pi) q[29];
U1q(0.367541807975701*pi,0.756473335358799*pi) q[30];
U1q(0.376445081094904*pi,1.732095237995648*pi) q[31];
U1q(0.64900853664332*pi,0.141794072585105*pi) q[32];
U1q(0.452655300887269*pi,1.00322121534088*pi) q[33];
U1q(0.239725407106139*pi,0.263220317492118*pi) q[34];
U1q(0.402385125847673*pi,0.233904602012959*pi) q[35];
U1q(0.619572245614804*pi,1.775182783781011*pi) q[36];
U1q(0.83766526644314*pi,1.05554506912343*pi) q[37];
U1q(0.0592107091250361*pi,1.725309065152038*pi) q[38];
U1q(0.615330115760559*pi,1.309771044790629*pi) q[39];
RZZ(0.0*pi) q[12],q[0];
RZZ(0.0*pi) q[1],q[14];
RZZ(0.0*pi) q[7],q[2];
RZZ(0.0*pi) q[39],q[3];
RZZ(0.0*pi) q[4],q[33];
RZZ(0.0*pi) q[30],q[5];
RZZ(0.0*pi) q[6],q[22];
RZZ(0.0*pi) q[8],q[16];
RZZ(0.0*pi) q[9],q[37];
RZZ(0.0*pi) q[11],q[10];
RZZ(0.0*pi) q[32],q[13];
RZZ(0.0*pi) q[15],q[35];
RZZ(0.0*pi) q[17],q[36];
RZZ(0.0*pi) q[34],q[18];
RZZ(0.0*pi) q[19],q[24];
RZZ(0.0*pi) q[20],q[23];
RZZ(0.0*pi) q[38],q[21];
RZZ(0.0*pi) q[25],q[31];
RZZ(0.0*pi) q[26],q[29];
RZZ(0.0*pi) q[27],q[28];
rz(2.98021105020966*pi) q[0];
rz(0.936842916597262*pi) q[1];
rz(1.11393016703116*pi) q[2];
rz(1.31477367859168*pi) q[3];
rz(2.74807909950344*pi) q[4];
rz(3.388339171103188*pi) q[5];
rz(1.37082225396733*pi) q[6];
rz(3.4940698466738658*pi) q[7];
rz(2.21555066338487*pi) q[8];
rz(3.9524548487958424*pi) q[9];
rz(2.4535258594442997*pi) q[10];
rz(0.876141440161787*pi) q[11];
rz(1.1883936036381*pi) q[12];
rz(1.66927688361291*pi) q[13];
rz(3.991711401506973*pi) q[14];
rz(1.73413737812445*pi) q[15];
rz(1.66415974621406*pi) q[16];
rz(2.7784141172533703*pi) q[17];
rz(0.404025027987523*pi) q[18];
rz(1.32970287467316*pi) q[19];
rz(0.294676695337204*pi) q[20];
rz(3.994060384546919*pi) q[21];
rz(2.02912601406171*pi) q[22];
rz(0.196663390155746*pi) q[23];
rz(1.27506138171962*pi) q[24];
rz(2.38439894473136*pi) q[25];
rz(3.378671028633388*pi) q[26];
rz(1.49808194470092*pi) q[27];
rz(1.57613644988617*pi) q[28];
rz(3.781236533792948*pi) q[29];
rz(0.825154174780798*pi) q[30];
rz(3.489525319621391*pi) q[31];
rz(0.931003533035893*pi) q[32];
rz(2.48723723154062*pi) q[33];
rz(1.13286713989884*pi) q[34];
rz(0.949720021128292*pi) q[35];
rz(2.38284937124143*pi) q[36];
rz(1.23568956752987*pi) q[37];
rz(1.02940726260179*pi) q[38];
rz(0.924185832324946*pi) q[39];
U1q(0.556901588029277*pi,1.649312301133403*pi) q[0];
U1q(0.491245364044351*pi,1.13730180361205*pi) q[1];
U1q(0.8949886991144*pi,0.797424833469265*pi) q[2];
U1q(0.118314103174671*pi,0.03086783931833*pi) q[3];
U1q(0.542968189445296*pi,1.243964356473769*pi) q[4];
U1q(0.780438907246457*pi,0.243562225918709*pi) q[5];
U1q(0.171188926331179*pi,0.566695530854309*pi) q[6];
U1q(0.671904038816211*pi,0.110117522674332*pi) q[7];
U1q(0.65685179510485*pi,1.285971139834464*pi) q[8];
U1q(0.29117809892726*pi,0.787270539095196*pi) q[9];
U1q(0.878489975599056*pi,1.876525340441042*pi) q[10];
U1q(0.309783123354709*pi,0.481556654202299*pi) q[11];
U1q(0.137725143865748*pi,0.146813196871818*pi) q[12];
U1q(0.503145114677222*pi,1.21214501590217*pi) q[13];
U1q(0.408084772204219*pi,1.219319304900836*pi) q[14];
U1q(0.680203572188031*pi,1.42993229003387*pi) q[15];
U1q(0.759790801893339*pi,1.54402602666474*pi) q[16];
U1q(0.563270789871465*pi,1.78190017746005*pi) q[17];
U1q(0.363666267156787*pi,1.488312351528423*pi) q[18];
U1q(0.187511560837483*pi,0.556640568667225*pi) q[19];
U1q(0.123254769521774*pi,1.423602700785688*pi) q[20];
U1q(0.418551878848374*pi,0.962176672961391*pi) q[21];
U1q(0.657765943597141*pi,1.094986840286414*pi) q[22];
U1q(0.57848392267036*pi,0.741321874944019*pi) q[23];
U1q(0.298802501237097*pi,1.773212733657184*pi) q[24];
U1q(0.617720163514395*pi,1.70390803581888*pi) q[25];
U1q(0.629991717347326*pi,1.844478364256008*pi) q[26];
U1q(0.205187516572864*pi,1.20450967745388*pi) q[27];
U1q(0.63799450638288*pi,1.25489324206467*pi) q[28];
U1q(0.115355436309126*pi,0.400900822512632*pi) q[29];
U1q(0.810929945352519*pi,0.299923132323258*pi) q[30];
U1q(0.817013992309939*pi,1.9689556870479286*pi) q[31];
U1q(0.764322646832485*pi,0.325833171216208*pi) q[32];
U1q(0.444897763827705*pi,1.8720992024352001*pi) q[33];
U1q(0.793775585279649*pi,0.718989562388392*pi) q[34];
U1q(0.958463629731314*pi,0.987062491531482*pi) q[35];
U1q(0.521841264899387*pi,1.377545205033205*pi) q[36];
U1q(0.501938066002915*pi,1.41435331158934*pi) q[37];
U1q(0.265790501799547*pi,1.31462743582892*pi) q[38];
U1q(0.470683249227746*pi,1.71052553170038*pi) q[39];
RZZ(0.0*pi) q[31],q[0];
RZZ(0.0*pi) q[9],q[1];
RZZ(0.0*pi) q[19],q[2];
RZZ(0.0*pi) q[3],q[36];
RZZ(0.0*pi) q[4],q[16];
RZZ(0.0*pi) q[5],q[33];
RZZ(0.0*pi) q[6],q[20];
RZZ(0.0*pi) q[7],q[23];
RZZ(0.0*pi) q[8],q[22];
RZZ(0.0*pi) q[13],q[10];
RZZ(0.0*pi) q[11],q[37];
RZZ(0.0*pi) q[35],q[12];
RZZ(0.0*pi) q[34],q[14];
RZZ(0.0*pi) q[15],q[30];
RZZ(0.0*pi) q[17],q[28];
RZZ(0.0*pi) q[25],q[18];
RZZ(0.0*pi) q[27],q[21];
RZZ(0.0*pi) q[24],q[29];
RZZ(0.0*pi) q[26],q[32];
RZZ(0.0*pi) q[39],q[38];
rz(0.553787303120298*pi) q[0];
rz(0.455251149120892*pi) q[1];
rz(2.08044613200192*pi) q[2];
rz(0.455954625107643*pi) q[3];
rz(3.148216124441936*pi) q[4];
rz(0.23337131830608016*pi) q[5];
rz(3.238010432825769*pi) q[6];
rz(2.50184710042415*pi) q[7];
rz(1.21623714079464*pi) q[8];
rz(2.918206611972*pi) q[9];
rz(0.0324329619489079*pi) q[10];
rz(3.948424602609258*pi) q[11];
rz(0.206854210851326*pi) q[12];
rz(1.12463842915173*pi) q[13];
rz(3.013670406430494*pi) q[14];
rz(0.7098147011424*pi) q[15];
rz(0.0110607725651959*pi) q[16];
rz(0.279596042749433*pi) q[17];
rz(1.51080134988206*pi) q[18];
rz(3.888353705770459*pi) q[19];
rz(0.0081512944848664*pi) q[20];
rz(1.99754284432568*pi) q[21];
rz(0.454340514023311*pi) q[22];
rz(3.578486759655302*pi) q[23];
rz(0.419990744335347*pi) q[24];
rz(3.325692527448231*pi) q[25];
rz(1.0197022989291402*pi) q[26];
rz(1.27330399840338*pi) q[27];
rz(0.897797292337568*pi) q[28];
rz(3.880526136550661*pi) q[29];
rz(1.83986240003601*pi) q[30];
rz(0.0271534423563347*pi) q[31];
rz(3.798558258523037*pi) q[32];
rz(3.101990144874852*pi) q[33];
rz(0.0975036210290666*pi) q[34];
rz(0.067532936465897*pi) q[35];
rz(3.893985388001001*pi) q[36];
rz(1.46658363962231*pi) q[37];
rz(3.803885654673382*pi) q[38];
rz(3.802977432890476*pi) q[39];
U1q(3.250710160808037*pi,1.43248400923012*pi) q[0];
U1q(3.252255160090982*pi,1.9381352258826*pi) q[1];
U1q(3.302430441022277*pi,1.687661121909386*pi) q[2];
U1q(3.812784181594203*pi,0.349685911627195*pi) q[3];
U1q(3.788584612989075*pi,0.7574838243442701*pi) q[4];
U1q(3.88947734352695*pi,0.28139966405777006*pi) q[5];
U1q(3.33917892920109*pi,0.275201792937973*pi) q[6];
U1q(3.25284093944729*pi,1.742912636291452*pi) q[7];
U1q(3.636085509304257*pi,0.017098436997304*pi) q[8];
U1q(3.1481049110796118*pi,0.7414651750172501*pi) q[9];
U1q(3.801398948154811*pi,1.9422997387438479*pi) q[10];
U1q(3.659445092619902*pi,1.23335819670799*pi) q[11];
U1q(3.698752822003218*pi,1.858241413941744*pi) q[12];
U1q(3.688868555722791*pi,1.00980427250536*pi) q[13];
U1q(3.643729654192642*pi,0.79729304307964*pi) q[14];
U1q(3.699841727529066*pi,0.149930176739041*pi) q[15];
U1q(3.259819996736043*pi,0.197807617369911*pi) q[16];
U1q(3.503101484785855*pi,0.0961575140440613*pi) q[17];
U1q(3.425155649724407*pi,0.3875396527597399*pi) q[18];
U1q(3.413696525328434*pi,1.642485655734125*pi) q[19];
U1q(3.331186857558721*pi,0.294847920143624*pi) q[20];
U1q(3.512333242551831*pi,1.00311885077555*pi) q[21];
U1q(3.560831145661683*pi,1.18156160864027*pi) q[22];
U1q(3.4265038445907923*pi,0.719512263112403*pi) q[23];
U1q(3.366818241340755*pi,1.372580463976647*pi) q[24];
U1q(3.808649575505549*pi,0.224972924030709*pi) q[25];
U1q(3.432634986606398*pi,0.5553693702475999*pi) q[26];
U1q(3.821772826437746*pi,0.455662268968189*pi) q[27];
U1q(3.348484862213709*pi,1.9525746612790733*pi) q[28];
U1q(3.3162894797826947*pi,1.9729118061102595*pi) q[29];
U1q(3.735291823304851*pi,1.63044402865007*pi) q[30];
U1q(3.26462262066814*pi,0.627126270650701*pi) q[31];
U1q(3.586203818003561*pi,1.731896222174047*pi) q[32];
U1q(3.4138110562412303*pi,0.595930863554776*pi) q[33];
U1q(3.4615772152903*pi,1.754531997506074*pi) q[34];
U1q(3.668727707902831*pi,0.25586578303766*pi) q[35];
U1q(3.870971945581423*pi,1.02183830710459*pi) q[36];
U1q(3.297573478695572*pi,0.948841152862584*pi) q[37];
U1q(3.53900440083421*pi,1.014449979761523*pi) q[38];
U1q(3.56953433446847*pi,1.600659520809116*pi) q[39];
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
