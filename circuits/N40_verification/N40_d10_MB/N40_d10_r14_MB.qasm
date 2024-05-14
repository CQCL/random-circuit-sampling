OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.66736568096551*pi,1.672606260979248*pi) q[0];
U1q(1.41588143973601*pi,1.4148179443967226*pi) q[1];
U1q(1.69856310575569*pi,1.559058201786651*pi) q[2];
U1q(0.437245473605979*pi,0.610704843543271*pi) q[3];
U1q(1.6463972533477*pi,0.8320578258712041*pi) q[4];
U1q(1.71765358738265*pi,1.8055212015773034*pi) q[5];
U1q(1.31871578946749*pi,0.7411334655510491*pi) q[6];
U1q(0.182291295835377*pi,0.9156341740579701*pi) q[7];
U1q(0.79456006285411*pi,1.8171343195232441*pi) q[8];
U1q(1.55351108205752*pi,1.1834274327165288*pi) q[9];
U1q(1.46259636793969*pi,0.5944589261364918*pi) q[10];
U1q(1.17100358522893*pi,1.9459712769900552*pi) q[11];
U1q(0.490993618434159*pi,0.253399330274184*pi) q[12];
U1q(0.659440326541863*pi,1.006097971863925*pi) q[13];
U1q(1.88721600148972*pi,0.9746175283531029*pi) q[14];
U1q(1.73510830739597*pi,1.651226538861588*pi) q[15];
U1q(1.57842074412744*pi,0.9812776626852899*pi) q[16];
U1q(1.10279867825069*pi,1.8918706192534978*pi) q[17];
U1q(0.347519216505594*pi,0.91984090549394*pi) q[18];
U1q(0.737994055193009*pi,1.303548157094264*pi) q[19];
U1q(0.472033669019371*pi,1.877902958797053*pi) q[20];
U1q(0.0250605463744081*pi,0.329012862272839*pi) q[21];
U1q(0.643789414877656*pi,0.687695431037343*pi) q[22];
U1q(1.64584373175693*pi,1.608109634659613*pi) q[23];
U1q(1.23832210781877*pi,1.6780491785178568*pi) q[24];
U1q(0.568526078316542*pi,1.9595026367391355*pi) q[25];
U1q(1.56233917060488*pi,1.1048492147053297*pi) q[26];
U1q(0.631920193868193*pi,1.4266003920672619*pi) q[27];
U1q(1.29569401556302*pi,0.5150660609303753*pi) q[28];
U1q(3.3757671198169747*pi,1.6179618976850094*pi) q[29];
U1q(1.76049104594069*pi,0.6110554110149792*pi) q[30];
U1q(1.22652550979892*pi,0.46877703101523605*pi) q[31];
U1q(0.336274585223673*pi,0.405295386700653*pi) q[32];
U1q(1.64061847007264*pi,0.6146089996111147*pi) q[33];
U1q(0.114813618637912*pi,1.9295491580245863*pi) q[34];
U1q(3.1432348561023717*pi,1.2351838912304947*pi) q[35];
U1q(1.91643602784536*pi,1.6262766656677399*pi) q[36];
U1q(1.66199405869842*pi,1.5302025537874986*pi) q[37];
U1q(1.16023128235268*pi,0.9826291585650528*pi) q[38];
U1q(1.57382843779697*pi,0.7089081124122335*pi) q[39];
RZZ(0.5*pi) q[24],q[0];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[2],q[21];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[5],q[22];
RZZ(0.5*pi) q[29],q[6];
RZZ(0.5*pi) q[34],q[7];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[10],q[13];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[33],q[17];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[23],q[31];
RZZ(0.5*pi) q[25],q[26];
RZZ(0.5*pi) q[30],q[38];
U1q(0.639373760821819*pi,0.78999998070582*pi) q[0];
U1q(0.738696982123144*pi,1.7179783224149023*pi) q[1];
U1q(0.465920913559816*pi,0.313065866159681*pi) q[2];
U1q(0.495274300061039*pi,1.9101557408411098*pi) q[3];
U1q(0.785907408353433*pi,0.29806390820252426*pi) q[4];
U1q(0.490702691725451*pi,0.3603453757633135*pi) q[5];
U1q(0.676343177220387*pi,1.217370069757189*pi) q[6];
U1q(0.135203710910223*pi,1.00057617908341*pi) q[7];
U1q(0.290201413263686*pi,0.09594171406569996*pi) q[8];
U1q(0.14388938704744*pi,1.8580792727019189*pi) q[9];
U1q(0.385979494906494*pi,1.774685543912562*pi) q[10];
U1q(0.820013312601052*pi,0.33257476783616546*pi) q[11];
U1q(0.84633708377801*pi,0.09238759546402009*pi) q[12];
U1q(0.562147493551356*pi,1.6506963569997204*pi) q[13];
U1q(0.0193767788232663*pi,1.9018155315520828*pi) q[14];
U1q(0.302859464991106*pi,0.7111394282110979*pi) q[15];
U1q(0.298991123883213*pi,1.4021514223617597*pi) q[16];
U1q(0.232822716642306*pi,1.905648890951098*pi) q[17];
U1q(0.804807109718975*pi,0.038478556116799956*pi) q[18];
U1q(0.467875475229899*pi,1.0806449001847498*pi) q[19];
U1q(0.756893731932225*pi,0.3329920145199601*pi) q[20];
U1q(0.282315825725029*pi,1.34670060821722*pi) q[21];
U1q(0.488207888052762*pi,1.1037129795535519*pi) q[22];
U1q(0.481961097965196*pi,1.6915843706771332*pi) q[23];
U1q(0.32121295900516*pi,1.9090119605402869*pi) q[24];
U1q(0.432931094893946*pi,0.45635817713580007*pi) q[25];
U1q(0.538425445819277*pi,0.7722027011130308*pi) q[26];
U1q(0.0923546804932396*pi,0.9251859572878902*pi) q[27];
U1q(0.968167108516847*pi,0.20192510002228525*pi) q[28];
U1q(0.356977634767518*pi,1.7275428484690294*pi) q[29];
U1q(0.506480438553838*pi,0.6208441172579793*pi) q[30];
U1q(0.347044596045587*pi,1.342436694295266*pi) q[31];
U1q(0.243012736796714*pi,0.8258858937956699*pi) q[32];
U1q(0.469661224813896*pi,1.2704130519344345*pi) q[33];
U1q(0.467930441304196*pi,0.39288607900827*pi) q[34];
U1q(0.298646506586064*pi,1.719791726503825*pi) q[35];
U1q(0.62688761558316*pi,1.1942132049829*pi) q[36];
U1q(0.17407917648119*pi,0.7238621700688688*pi) q[37];
U1q(0.116446908774868*pi,1.1458698488350227*pi) q[38];
U1q(0.139054381180891*pi,0.3932700507055733*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[21],q[3];
RZZ(0.5*pi) q[4],q[31];
RZZ(0.5*pi) q[5],q[35];
RZZ(0.5*pi) q[6],q[26];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[8],q[13];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[15],q[14];
RZZ(0.5*pi) q[16],q[24];
RZZ(0.5*pi) q[19],q[17];
RZZ(0.5*pi) q[30],q[18];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[25],q[33];
RZZ(0.5*pi) q[27],q[34];
U1q(0.259864545286509*pi,0.4194177888906099*pi) q[0];
U1q(0.379723384438828*pi,0.626168339722863*pi) q[1];
U1q(0.814120503642136*pi,1.5096390158601407*pi) q[2];
U1q(0.744276574410507*pi,0.8603117620332403*pi) q[3];
U1q(0.839730885711442*pi,1.165756926426234*pi) q[4];
U1q(0.396718313957631*pi,0.4378702467927429*pi) q[5];
U1q(0.774850147164305*pi,0.5830343262326387*pi) q[6];
U1q(0.137812635078476*pi,0.33701319766993976*pi) q[7];
U1q(0.326072069113765*pi,0.1845161167552698*pi) q[8];
U1q(0.489749191329929*pi,1.6700645724595784*pi) q[9];
U1q(0.616662908814783*pi,1.3893939719615518*pi) q[10];
U1q(0.566974642543203*pi,0.27013426411051533*pi) q[11];
U1q(0.679688574041651*pi,0.91764177607494*pi) q[12];
U1q(0.500926201593494*pi,0.38485296528071*pi) q[13];
U1q(0.200915079634104*pi,1.1437239479384127*pi) q[14];
U1q(0.627999665199987*pi,1.864196788842908*pi) q[15];
U1q(0.576436730362631*pi,0.58499407741302*pi) q[16];
U1q(0.635354186726184*pi,0.05345067798043779*pi) q[17];
U1q(0.377904123050746*pi,0.9695685446910298*pi) q[18];
U1q(0.612328832063583*pi,1.5314037273606704*pi) q[19];
U1q(0.737314782849279*pi,1.8382291025559399*pi) q[20];
U1q(0.208164262410635*pi,0.9434224730795604*pi) q[21];
U1q(0.404669004607025*pi,0.05152276920082022*pi) q[22];
U1q(0.545572408306201*pi,1.4840671878735323*pi) q[23];
U1q(0.527458239555063*pi,0.6466309253068774*pi) q[24];
U1q(0.565909340379141*pi,1.8485840115562704*pi) q[25];
U1q(0.864572565074424*pi,1.5570174777442598*pi) q[26];
U1q(0.124286303021258*pi,0.07333324798278973*pi) q[27];
U1q(0.31054527094677*pi,1.4872223774701956*pi) q[28];
U1q(0.757522420899748*pi,0.9179470409702297*pi) q[29];
U1q(0.27391092503878*pi,0.2317085550925686*pi) q[30];
U1q(0.446539222269293*pi,0.9688762895956957*pi) q[31];
U1q(0.469320768103307*pi,1.8070846014295796*pi) q[32];
U1q(0.567088597406671*pi,0.8826879094661346*pi) q[33];
U1q(0.35213313928936*pi,0.5039597862061704*pi) q[34];
U1q(0.108050692567005*pi,0.7394002625777452*pi) q[35];
U1q(0.544262761274321*pi,1.5687340185640606*pi) q[36];
U1q(0.300575974296347*pi,0.31727259177543843*pi) q[37];
U1q(0.303650293574562*pi,1.7602254103360728*pi) q[38];
U1q(0.4276341939511*pi,0.1287829540792531*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[4],q[33];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[17];
RZZ(0.5*pi) q[30],q[7];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[16],q[9];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[25],q[13];
RZZ(0.5*pi) q[18],q[24];
RZZ(0.5*pi) q[19],q[37];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[32],q[26];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[34],q[28];
RZZ(0.5*pi) q[35],q[36];
U1q(0.417251832423225*pi,0.8327092616161202*pi) q[0];
U1q(0.447130037541168*pi,1.6711332591468127*pi) q[1];
U1q(0.710051290969052*pi,1.5503617517212112*pi) q[2];
U1q(0.448811849758583*pi,1.1219015852225596*pi) q[3];
U1q(0.156523708080928*pi,0.8029544915493538*pi) q[4];
U1q(0.361207373543904*pi,1.0376902570027529*pi) q[5];
U1q(0.775633555878956*pi,0.9349702110859486*pi) q[6];
U1q(0.480240483996385*pi,0.3230918501839497*pi) q[7];
U1q(0.537589483893245*pi,0.8200707939505696*pi) q[8];
U1q(0.639353678694999*pi,1.0651308392664696*pi) q[9];
U1q(0.734921630659698*pi,1.1967836297078511*pi) q[10];
U1q(0.511483341910625*pi,1.013842645355255*pi) q[11];
U1q(0.796983494584948*pi,0.01918797801052019*pi) q[12];
U1q(0.587017931954212*pi,1.5998449745664702*pi) q[13];
U1q(0.204070762759928*pi,1.9180945803365832*pi) q[14];
U1q(0.174847001743566*pi,1.6618953593574783*pi) q[15];
U1q(0.701767225034867*pi,1.63182644896564*pi) q[16];
U1q(0.40113307628742*pi,1.9995993019712879*pi) q[17];
U1q(0.378509450396902*pi,0.9065211847514796*pi) q[18];
U1q(0.461179741735391*pi,1.4455761770894204*pi) q[19];
U1q(0.691231182140926*pi,1.0663460620044898*pi) q[20];
U1q(0.676059897322575*pi,1.3010248895266496*pi) q[21];
U1q(0.333538351981856*pi,0.11316460282422991*pi) q[22];
U1q(0.291804339704185*pi,1.7485800157508224*pi) q[23];
U1q(0.557858528705115*pi,0.9708632539794975*pi) q[24];
U1q(0.547327096027824*pi,0.9974684357920696*pi) q[25];
U1q(0.651488072555816*pi,0.11259645089783987*pi) q[26];
U1q(0.453345416660494*pi,0.23642406695957963*pi) q[27];
U1q(0.369194470923459*pi,0.9046211762719958*pi) q[28];
U1q(0.39771251748532*pi,0.4799719976890193*pi) q[29];
U1q(0.285043336422818*pi,0.6255261021869689*pi) q[30];
U1q(0.647887451850003*pi,1.6630126329860655*pi) q[31];
U1q(0.812817311305095*pi,0.6149566021095003*pi) q[32];
U1q(0.484023751966868*pi,1.4921380609563144*pi) q[33];
U1q(0.601713931181148*pi,0.00876740971699963*pi) q[34];
U1q(0.213465047825073*pi,1.8115120395907347*pi) q[35];
U1q(0.661559602071861*pi,1.22312019978758*pi) q[36];
U1q(0.470310943147028*pi,1.970455768716059*pi) q[37];
U1q(0.421878734081511*pi,0.7343815292618432*pi) q[38];
U1q(0.651424573057825*pi,1.9574552170876238*pi) q[39];
RZZ(0.5*pi) q[38],q[0];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[2],q[9];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[15],q[4];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[8],q[23];
RZZ(0.5*pi) q[12],q[31];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[19],q[14];
RZZ(0.5*pi) q[16],q[35];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[20],q[34];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[25],q[32];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[27],q[33];
U1q(0.483409133458959*pi,1.1791029341412003*pi) q[0];
U1q(0.671464116362046*pi,1.6439481718614015*pi) q[1];
U1q(0.924972756077179*pi,1.1803419229629615*pi) q[2];
U1q(0.214066029916214*pi,0.5869954828980006*pi) q[3];
U1q(0.509143509294657*pi,1.5151834588021238*pi) q[4];
U1q(0.327127679031244*pi,0.00308566183250214*pi) q[5];
U1q(0.766103503519645*pi,1.9030836310377293*pi) q[6];
U1q(0.392273440092608*pi,0.7562454403246992*pi) q[7];
U1q(0.355856945825439*pi,1.5250274080980795*pi) q[8];
U1q(0.916445462118708*pi,1.666835952927558*pi) q[9];
U1q(0.666976687530789*pi,0.858661516260522*pi) q[10];
U1q(0.604783382124089*pi,0.745150316609756*pi) q[11];
U1q(0.906120281555326*pi,1.4436747309692004*pi) q[12];
U1q(0.54370820024426*pi,0.13267933268228038*pi) q[13];
U1q(0.358411390148732*pi,0.6694563197783436*pi) q[14];
U1q(0.156066552802355*pi,1.1398177357346881*pi) q[15];
U1q(0.803827857041422*pi,0.5371672957330809*pi) q[16];
U1q(0.615414847321942*pi,0.9538093568846975*pi) q[17];
U1q(0.391837638245949*pi,1.3683593790919204*pi) q[18];
U1q(0.436664437743799*pi,1.2043992873915492*pi) q[19];
U1q(0.791645143935725*pi,0.6684437127780694*pi) q[20];
U1q(0.82010424531014*pi,1.41865041293992*pi) q[21];
U1q(0.656455412518537*pi,1.0444329517651596*pi) q[22];
U1q(0.789488910325568*pi,0.7762389944624726*pi) q[23];
U1q(0.606886732169794*pi,0.16889796088249742*pi) q[24];
U1q(0.230441918013959*pi,0.07479389711654072*pi) q[25];
U1q(0.796364614204612*pi,1.0824754792958498*pi) q[26];
U1q(0.721546817158144*pi,1.2145032947708998*pi) q[27];
U1q(0.286146828235802*pi,1.209717071528745*pi) q[28];
U1q(0.812324507669749*pi,1.4819737419999495*pi) q[29];
U1q(0.353989642166344*pi,1.870662026946178*pi) q[30];
U1q(0.86408926363648*pi,1.2795291936486368*pi) q[31];
U1q(0.214412249846218*pi,1.5294416254660002*pi) q[32];
U1q(0.565105867071097*pi,1.4009620576841044*pi) q[33];
U1q(0.68019742025445*pi,0.08491112852166971*pi) q[34];
U1q(0.269220492862415*pi,1.7142148135883648*pi) q[35];
U1q(0.690407146667564*pi,1.9719826948525512*pi) q[36];
U1q(0.547992810445105*pi,0.8401197467002994*pi) q[37];
U1q(0.7395773209107*pi,0.9359387040103124*pi) q[38];
U1q(0.13608614985291*pi,1.3012175751501438*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[2],q[36];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[5],q[31];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[39],q[8];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[24],q[12];
RZZ(0.5*pi) q[22],q[13];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[18],q[19];
RZZ(0.5*pi) q[20],q[27];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[38],q[37];
U1q(0.742610193179419*pi,1.1518571951864*pi) q[0];
U1q(0.334643455430694*pi,0.5136998347042212*pi) q[1];
U1q(0.819810602881651*pi,1.7365797911140515*pi) q[2];
U1q(0.912331549078422*pi,1.7926113205903995*pi) q[3];
U1q(0.621969808502508*pi,0.014512669962704727*pi) q[4];
U1q(0.683164446147218*pi,1.5771903626191026*pi) q[5];
U1q(0.95404773726738*pi,1.7688946908323402*pi) q[6];
U1q(0.84456031118824*pi,0.4517086675037998*pi) q[7];
U1q(0.689006345268965*pi,0.9671126713309004*pi) q[8];
U1q(0.254376450988923*pi,0.9212544514161394*pi) q[9];
U1q(0.935880647377962*pi,1.0122679248266913*pi) q[10];
U1q(0.502243629002047*pi,1.6018096985603556*pi) q[11];
U1q(0.264320413810353*pi,0.7753094880795004*pi) q[12];
U1q(0.863204000432261*pi,0.5178663851145107*pi) q[13];
U1q(0.333560271416297*pi,0.9862884133070011*pi) q[14];
U1q(0.455974908286985*pi,1.9091778993611879*pi) q[15];
U1q(0.669784329752395*pi,0.7564420480479903*pi) q[16];
U1q(0.620029226633363*pi,0.985512984548297*pi) q[17];
U1q(0.345635819055843*pi,0.12894480189386037*pi) q[18];
U1q(0.955744952903984*pi,1.3491811175620292*pi) q[19];
U1q(0.117945192903168*pi,1.9554575948096993*pi) q[20];
U1q(0.773312768376078*pi,0.7251383534187994*pi) q[21];
U1q(0.530348338113297*pi,1.6147230625196993*pi) q[22];
U1q(0.296408258564726*pi,1.377673589073213*pi) q[23];
U1q(0.527381274880682*pi,0.10504642004269549*pi) q[24];
U1q(0.122067827258217*pi,1.4882558673771005*pi) q[25];
U1q(0.571655790312087*pi,1.8652170646412998*pi) q[26];
U1q(0.838340168002866*pi,0.5164867029993001*pi) q[27];
U1q(0.479972744015539*pi,1.441887608894275*pi) q[28];
U1q(0.953011388106958*pi,0.5843177341612105*pi) q[29];
U1q(0.861720808227721*pi,0.5043700222001792*pi) q[30];
U1q(0.254793313924001*pi,0.29583833059493614*pi) q[31];
U1q(0.853643826322342*pi,0.11178874562247021*pi) q[32];
U1q(0.251027926474719*pi,0.3355378909697144*pi) q[33];
U1q(0.125325843688066*pi,1.1445097986429005*pi) q[34];
U1q(0.715016542682956*pi,0.9491451459477958*pi) q[35];
U1q(0.331663590551159*pi,1.7465801117359412*pi) q[36];
U1q(0.331832762388118*pi,1.171665669705499*pi) q[37];
U1q(0.050588683580409*pi,1.773533388143992*pi) q[38];
U1q(0.216685214871554*pi,0.7349874207290341*pi) q[39];
rz(0.20027318678440054*pi) q[0];
rz(3.5973291903162785*pi) q[1];
rz(0.7695139138407487*pi) q[2];
rz(2.397360106299301*pi) q[3];
rz(0.08669406017639503*pi) q[4];
rz(0.3505982567915975*pi) q[5];
rz(1.8045118657625494*pi) q[6];
rz(0.6941879598523997*pi) q[7];
rz(2.8892400400261007*pi) q[8];
rz(0.5515086223591119*pi) q[9];
rz(2.4814576738014082*pi) q[10];
rz(2.5358889917139447*pi) q[11];
rz(2.30460087028095*pi) q[12];
rz(1.32125630983629*pi) q[13];
rz(2.881043727554296*pi) q[14];
rz(0.28118648278361213*pi) q[15];
rz(1.4934113982928103*pi) q[16];
rz(1.645479142446403*pi) q[17];
rz(2.365244648031201*pi) q[18];
rz(0.9313832825600006*pi) q[19];
rz(1.0160238578122005*pi) q[20];
rz(0.6941810625211993*pi) q[21];
rz(0.9993037111189*pi) q[22];
rz(3.7314572733591866*pi) q[23];
rz(2.1828011381298538*pi) q[24];
rz(1.3963043483077993*pi) q[25];
rz(2.514132871514919*pi) q[26];
rz(1.9761199299931995*pi) q[27];
rz(3.4553923178760257*pi) q[28];
rz(0.6717969724947892*pi) q[29];
rz(2.555817677947422*pi) q[30];
rz(0.8240319230820639*pi) q[31];
rz(3.22923358955215*pi) q[32];
rz(1.5917889127429863*pi) q[33];
rz(1.4009180657646993*pi) q[34];
rz(2.590979150356585*pi) q[35];
rz(0.28786482776645883*pi) q[36];
rz(2.874390505177402*pi) q[37];
rz(1.2341556908111464*pi) q[38];
rz(2.0711486645702664*pi) q[39];
U1q(0.742610193179419*pi,0.352130381970792*pi) q[0];
U1q(1.33464345543069*pi,1.111029025020497*pi) q[1];
U1q(1.81981060288165*pi,1.506093704954738*pi) q[2];
U1q(0.912331549078422*pi,1.189971426889761*pi) q[3];
U1q(1.62196980850251*pi,1.1012067301391*pi) q[4];
U1q(1.68316444614722*pi,0.927788619410664*pi) q[5];
U1q(1.95404773726738*pi,0.573406556594937*pi) q[6];
U1q(0.84456031118824*pi,0.145896627356224*pi) q[7];
U1q(1.68900634526897*pi,0.856352711357022*pi) q[8];
U1q(3.2543764509889233*pi,0.472763073775257*pi) q[9];
U1q(0.935880647377962*pi,0.493725598628094*pi) q[10];
U1q(3.5022436290020478*pi,1.137698690274308*pi) q[11];
U1q(1.26432041381035*pi,0.0799103583604485*pi) q[12];
U1q(0.863204000432261*pi,0.83912269495079*pi) q[13];
U1q(0.333560271416297*pi,0.86733214086123*pi) q[14];
U1q(1.45597490828698*pi,1.190364382144807*pi) q[15];
U1q(3.669784329752395*pi,1.249853446340857*pi) q[16];
U1q(0.620029226633363*pi,1.630992126994737*pi) q[17];
U1q(1.34563581905584*pi,1.494189449925034*pi) q[18];
U1q(0.955744952903984*pi,1.28056440012202*pi) q[19];
U1q(1.11794519290317*pi,1.971481452621915*pi) q[20];
U1q(1.77331276837608*pi,0.419319415940059*pi) q[21];
U1q(0.530348338113297*pi,1.614026773638624*pi) q[22];
U1q(1.29640825856473*pi,0.109130862432337*pi) q[23];
U1q(1.52738127488068*pi,1.287847558172551*pi) q[24];
U1q(0.122067827258217*pi,1.884560215684866*pi) q[25];
U1q(1.57165579031209*pi,1.379349936156222*pi) q[26];
U1q(0.838340168002866*pi,1.492606632992481*pi) q[27];
U1q(3.47997274401554*pi,1.897279926770312*pi) q[28];
U1q(1.95301138810696*pi,0.256114706655925*pi) q[29];
U1q(0.861720808227721*pi,0.0601877001476304*pi) q[30];
U1q(0.254793313924001*pi,0.119870253677039*pi) q[31];
U1q(0.853643826322342*pi,0.341022335174622*pi) q[32];
U1q(1.25102792647472*pi,0.927326803712729*pi) q[33];
U1q(1.12532584368807*pi,1.545427864407628*pi) q[34];
U1q(3.715016542682956*pi,0.540124296304366*pi) q[35];
U1q(0.331663590551159*pi,1.034444939502394*pi) q[36];
U1q(0.331832762388118*pi,1.046056174882805*pi) q[37];
U1q(3.050588683580409*pi,0.00768907895513637*pi) q[38];
U1q(0.216685214871554*pi,1.806136085299302*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[2],q[36];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[5],q[31];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[39],q[8];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[24],q[12];
RZZ(0.5*pi) q[22],q[13];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[18],q[19];
RZZ(0.5*pi) q[20],q[27];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[38],q[37];
U1q(0.483409133458959*pi,0.379376120925586*pi) q[0];
U1q(3.671464116362046*pi,1.9807806878632745*pi) q[1];
U1q(3.07502724392282*pi,1.0623315731058036*pi) q[2];
U1q(1.21406602991621*pi,0.9843555891973699*pi) q[3];
U1q(1.50914350929466*pi,1.600535941299674*pi) q[4];
U1q(3.672872320968756*pi,1.501893320197217*pi) q[5];
U1q(1.76610350351965*pi,1.4392176163895452*pi) q[6];
U1q(0.392273440092608*pi,0.450433400177088*pi) q[7];
U1q(1.35585694582544*pi,0.29843797458985655*pi) q[8];
U1q(3.083554537881292*pi,0.7271815722638415*pi) q[9];
U1q(1.66697668753079*pi,0.34011919006192004*pi) q[10];
U1q(1.60478338212409*pi,0.994358072224879*pi) q[11];
U1q(1.90612028155533*pi,0.411545115470751*pi) q[12];
U1q(0.54370820024426*pi,1.4539356425185601*pi) q[13];
U1q(1.35841139014873*pi,0.5505000473326098*pi) q[14];
U1q(3.843933447197644*pi,0.9597245457713424*pi) q[15];
U1q(1.80382785704142*pi,0.46912819865577715*pi) q[16];
U1q(1.61541484732194*pi,1.599288499331105*pi) q[17];
U1q(3.608162361754051*pi,1.2547748727269745*pi) q[18];
U1q(1.4366644377438*pi,1.135782569951542*pi) q[19];
U1q(3.208354856064277*pi,0.2584953346535821*pi) q[20];
U1q(3.82010424531014*pi,1.725807356418986*pi) q[21];
U1q(1.65645541251854*pi,0.04373666288409006*pi) q[22];
U1q(1.78948891032557*pi,1.7105654570430529*pi) q[23];
U1q(1.60688673216979*pi,0.2239960173327593*pi) q[24];
U1q(0.230441918013959*pi,0.471098245424338*pi) q[25];
U1q(1.79636461420461*pi,1.1620915215016656*pi) q[26];
U1q(0.721546817158144*pi,0.19062322476407*pi) q[27];
U1q(3.713853171764198*pi,1.1294504641358394*pi) q[28];
U1q(1.81232450766975*pi,0.35845869881715664*pi) q[29];
U1q(1.35398964216634*pi,1.426479704893618*pi) q[30];
U1q(3.864089263636481*pi,1.1035611167307202*pi) q[31];
U1q(1.21441224984622*pi,0.7586752150181499*pi) q[32];
U1q(1.5651058670711*pi,0.8619026369983573*pi) q[33];
U1q(3.31980257974555*pi,0.6050265345288814*pi) q[34];
U1q(1.26922049286242*pi,1.7750546286637874*pi) q[35];
U1q(0.690407146667564*pi,1.259847522619033*pi) q[36];
U1q(0.547992810445105*pi,0.7145102518776201*pi) q[37];
U1q(1.7395773209107*pi,0.8452837630888206*pi) q[38];
U1q(0.13608614985291*pi,0.37236623972037*pi) q[39];
RZZ(0.5*pi) q[38],q[0];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[2],q[9];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[15],q[4];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[8],q[23];
RZZ(0.5*pi) q[12],q[31];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[19],q[14];
RZZ(0.5*pi) q[16],q[35];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[20],q[34];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[25],q[32];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[27],q[33];
U1q(1.41725183242322*pi,1.032982448400516*pi) q[0];
U1q(1.44713003754117*pi,1.0079657751486772*pi) q[1];
U1q(1.71005129096905*pi,0.6923117443475497*pi) q[2];
U1q(3.551188150241417*pi,1.4494494868728403*pi) q[3];
U1q(0.156523708080928*pi,0.8883069740469041*pi) q[4];
U1q(1.3612073735439*pi,0.4672887250269677*pi) q[5];
U1q(0.775633555878956*pi,1.4711041964377585*pi) q[6];
U1q(1.48024048399639*pi,1.01727981003638*pi) q[7];
U1q(0.537589483893245*pi,0.5934813604423466*pi) q[8];
U1q(3.639353678695*pi,1.3288866859249289*pi) q[9];
U1q(3.2650783693403023*pi,0.001997076614593807*pi) q[10];
U1q(0.511483341910625*pi,0.2630504009703789*pi) q[11];
U1q(0.796983494584948*pi,0.987058362512073*pi) q[12];
U1q(0.587017931954212*pi,0.92110128440275*pi) q[13];
U1q(3.795929237240072*pi,1.3018617867743725*pi) q[14];
U1q(1.17484700174357*pi,1.4376469221485573*pi) q[15];
U1q(1.70176722503487*pi,0.5637873518883376*pi) q[16];
U1q(3.59886692371258*pi,0.5534985542445168*pi) q[17];
U1q(3.621490549603097*pi,0.7166130670674136*pi) q[18];
U1q(3.538820258264608*pi,0.8946056802536637*pi) q[19];
U1q(3.308768817859074*pi,0.8605929854271652*pi) q[20];
U1q(3.676059897322575*pi,0.6081818330057231*pi) q[21];
U1q(1.33353835198186*pi,1.9750050118250226*pi) q[22];
U1q(0.291804339704185*pi,1.6829064783313958*pi) q[23];
U1q(1.55785852870511*pi,0.025961310429769124*pi) q[24];
U1q(1.54732709602782*pi,1.3937727840998662*pi) q[25];
U1q(1.65148807255582*pi,1.1922124931036557*pi) q[26];
U1q(1.45334541666049*pi,0.21254399695276982*pi) q[27];
U1q(3.630805529076541*pi,0.43454635939257924*pi) q[28];
U1q(0.39771251748532*pi,0.3564569545062315*pi) q[29];
U1q(3.714956663577182*pi,1.6716156296528324*pi) q[30];
U1q(1.64788745185*pi,0.7200776773932955*pi) q[31];
U1q(3.187182688694905*pi,0.673160238374646*pi) q[32];
U1q(1.48402375196687*pi,1.9530786402705687*pi) q[33];
U1q(3.398286068818852*pi,1.6811702533335513*pi) q[34];
U1q(1.21346504782507*pi,0.8723518546661584*pi) q[35];
U1q(0.661559602071861*pi,0.510985027554064*pi) q[36];
U1q(1.47031094314703*pi,0.8448462738934199*pi) q[37];
U1q(1.42187873408151*pi,1.6437265883403542*pi) q[38];
U1q(0.651424573057825*pi,0.028603881657850216*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[4],q[33];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[17];
RZZ(0.5*pi) q[30],q[7];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[16],q[9];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[25],q[13];
RZZ(0.5*pi) q[18],q[24];
RZZ(0.5*pi) q[19],q[37];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[32],q[26];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[34],q[28];
RZZ(0.5*pi) q[35],q[36];
U1q(3.740135454713491*pi,1.4462739211260245*pi) q[0];
U1q(1.37972338443883*pi,1.052930694572626*pi) q[1];
U1q(0.814120503642136*pi,1.6515890084864795*pi) q[2];
U1q(1.74427657441051*pi,1.7110393100621675*pi) q[3];
U1q(0.839730885711442*pi,1.2511094089237842*pi) q[4];
U1q(0.396718313957631*pi,0.8674687148169609*pi) q[5];
U1q(0.774850147164305*pi,1.1191683115844482*pi) q[6];
U1q(3.862187364921524*pi,1.0033584625503966*pi) q[7];
U1q(1.32607206911376*pi,0.9579266832470448*pi) q[8];
U1q(1.48974919132993*pi,1.933820419118039*pi) q[9];
U1q(1.61666290881478*pi,0.8093867343608943*pi) q[10];
U1q(1.5669746425432*pi,1.5193420197256389*pi) q[11];
U1q(0.679688574041651*pi,0.8855121605764831*pi) q[12];
U1q(1.50092620159349*pi,1.706109275117*pi) q[13];
U1q(3.799084920365896*pi,1.076232419172543*pi) q[14];
U1q(1.62799966519999*pi,0.6399483516339777*pi) q[15];
U1q(3.423563269637369*pi,0.61061972344097*pi) q[16];
U1q(1.63535418672618*pi,1.4996471782353673*pi) q[17];
U1q(3.622095876949254*pi,1.6535657071278687*pi) q[18];
U1q(1.61232883206358*pi,0.8087781299824215*pi) q[19];
U1q(3.262685217150721*pi,1.0887099448757254*pi) q[20];
U1q(3.208164262410635*pi,1.9657842494528248*pi) q[21];
U1q(1.40466900460702*pi,0.9133631782016227*pi) q[22];
U1q(1.5455724083062*pi,1.4183936504541155*pi) q[23];
U1q(3.472541760444937*pi,0.35019363910238965*pi) q[24];
U1q(1.56590934037914*pi,0.5426572083356653*pi) q[25];
U1q(3.135427434925576*pi,1.7477914662572318*pi) q[26];
U1q(1.12428630302126*pi,0.3756348159295664*pi) q[27];
U1q(3.68945472905323*pi,0.8519451581943902*pi) q[28];
U1q(0.757522420899748*pi,0.7944319977874317*pi) q[29];
U1q(1.27391092503878*pi,0.06543317674723614*pi) q[30];
U1q(3.446539222269293*pi,1.0259413340029262*pi) q[31];
U1q(1.46932076810331*pi,1.4810322390545614*pi) q[32];
U1q(3.432911402593329*pi,1.5625287917607453*pi) q[33];
U1q(3.64786686071064*pi,0.1859778768443816*pi) q[34];
U1q(1.108050692567*pi,1.9444636316791495*pi) q[35];
U1q(1.54426276127432*pi,0.85659884633055*pi) q[36];
U1q(3.300575974296347*pi,1.4980294508340437*pi) q[37];
U1q(3.696349706425438*pi,1.6178827072661424*pi) q[38];
U1q(1.4276341939511*pi,1.1999316186494804*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[21],q[3];
RZZ(0.5*pi) q[4],q[31];
RZZ(0.5*pi) q[5],q[35];
RZZ(0.5*pi) q[6],q[26];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[8],q[13];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[15],q[14];
RZZ(0.5*pi) q[16],q[24];
RZZ(0.5*pi) q[19],q[17];
RZZ(0.5*pi) q[30],q[18];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[25],q[33];
RZZ(0.5*pi) q[27],q[34];
U1q(3.360626239178181*pi,0.07569172931082746*pi) q[0];
U1q(0.738696982123144*pi,0.144740677264676*pi) q[1];
U1q(3.465920913559816*pi,0.45501585878602957*pi) q[2];
U1q(0.495274300061039*pi,1.760883288870037*pi) q[3];
U1q(0.785907408353433*pi,0.38341639070007405*pi) q[4];
U1q(0.490702691725451*pi,1.7899438437875377*pi) q[5];
U1q(0.676343177220387*pi,1.7535040551089986*pi) q[6];
U1q(3.864796289089777*pi,1.3397954811369273*pi) q[7];
U1q(3.709798586736314*pi,1.046501085936601*pi) q[8];
U1q(1.14388938704744*pi,1.7458057188756944*pi) q[9];
U1q(1.38597949490649*pi,1.1946783063119049*pi) q[10];
U1q(3.179986687398949*pi,0.45690151599999407*pi) q[11];
U1q(0.84633708377801*pi,1.0602579799655727*pi) q[12];
U1q(1.56214749355136*pi,1.4402658833979907*pi) q[13];
U1q(3.980623221176729*pi,1.3181408355588733*pi) q[14];
U1q(3.697140535008894*pi,1.793005712265785*pi) q[15];
U1q(3.298991123883213*pi,1.7934623784922188*pi) q[16];
U1q(0.232822716642306*pi,0.3518453912060373*pi) q[17];
U1q(1.80480710971897*pi,1.584655695702097*pi) q[18];
U1q(0.467875475229899*pi,1.3580193028065013*pi) q[19];
U1q(3.243106268067775*pi,1.5939470329116956*pi) q[20];
U1q(3.282315825725029*pi,1.3690623845904852*pi) q[21];
U1q(3.511792111947238*pi,0.861172967848896*pi) q[22];
U1q(1.4819610979652*pi,1.2108764676505168*pi) q[23];
U1q(3.67878704099484*pi,0.08781260386897038*pi) q[24];
U1q(0.432931094893946*pi,1.1504313739151966*pi) q[25];
U1q(1.53842544581928*pi,1.5326062428884644*pi) q[26];
U1q(3.092354680493239*pi,1.227487525234677*pi) q[27];
U1q(3.031832891483154*pi,1.1372424356422972*pi) q[28];
U1q(0.356977634767518*pi,0.6040278052862318*pi) q[29];
U1q(1.50648043855384*pi,0.45456873891265515*pi) q[30];
U1q(3.652955403954413*pi,1.6523809293033551*pi) q[31];
U1q(0.243012736796714*pi,0.49983353142065123*pi) q[32];
U1q(1.4696612248139*pi,0.17480364929244452*pi) q[33];
U1q(1.4679304413042*pi,1.297051584042276*pi) q[34];
U1q(1.29864650658606*pi,1.9248550956052326*pi) q[35];
U1q(3.37311238441684*pi,1.2311196599117133*pi) q[36];
U1q(0.17407917648119*pi,1.9046190291274634*pi) q[37];
U1q(3.883553091225132*pi,1.2322382687671825*pi) q[38];
U1q(3.860945618819109*pi,1.9354445220231646*pi) q[39];
RZZ(0.5*pi) q[24],q[0];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[2],q[21];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[5],q[22];
RZZ(0.5*pi) q[29],q[6];
RZZ(0.5*pi) q[34],q[7];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[10],q[13];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[33],q[17];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[23],q[31];
RZZ(0.5*pi) q[25],q[26];
RZZ(0.5*pi) q[30],q[38];
U1q(3.66736568096551*pi,1.1930854490373957*pi) q[0];
U1q(0.415881439736014*pi,1.8415802992464956*pi) q[1];
U1q(1.69856310575569*pi,0.20902352315906514*pi) q[2];
U1q(0.437245473605979*pi,0.46143239157220783*pi) q[3];
U1q(0.646397253347697*pi,0.9174103083687539*pi) q[4];
U1q(0.717653587382651*pi,1.2351196696015174*pi) q[5];
U1q(0.318715789467491*pi,0.2772674509028583*pi) q[6];
U1q(1.18229129583538*pi,0.4247374861623685*pi) q[7];
U1q(1.79456006285411*pi,0.32530848047906424*pi) q[8];
U1q(0.553511082057522*pi,1.0711538788903003*pi) q[9];
U1q(1.46259636793969*pi,0.3749049240879776*pi) q[10];
U1q(1.17100358522893*pi,0.8435050068461019*pi) q[11];
U1q(0.490993618434159*pi,0.2212697147757332*pi) q[12];
U1q(0.659440326541863*pi,0.7956674982621907*pi) q[13];
U1q(1.88721600148972*pi,1.2453388387578546*pi) q[14];
U1q(1.73510830739597*pi,1.8529186016152992*pi) q[15];
U1q(0.578420744127442*pi,1.3725886188157492*pi) q[16];
U1q(0.102798678250686*pi,1.3380671195084268*pi) q[17];
U1q(0.347519216505594*pi,0.46601804507922706*pi) q[18];
U1q(0.737994055193009*pi,1.5809225597160212*pi) q[19];
U1q(3.472033669019371*pi,1.049036088634602*pi) q[20];
U1q(1.02506054637441*pi,1.3867501305348715*pi) q[21];
U1q(3.643789414877656*pi,0.2771905163650983*pi) q[22];
U1q(0.645843731756926*pi,1.127401731632987*pi) q[23];
U1q(1.23832210781877*pi,0.31877538589140464*pi) q[24];
U1q(0.568526078316542*pi,1.6535758335185289*pi) q[25];
U1q(0.562339170604876*pi,0.865252756480765*pi) q[26];
U1q(1.63192019386819*pi,0.7260730904553077*pi) q[27];
U1q(1.29569401556302*pi,0.8241014747342033*pi) q[28];
U1q(0.375767119816975*pi,1.494446854502212*pi) q[29];
U1q(1.76049104594069*pi,1.464357445155656*pi) q[30];
U1q(1.22652550979892*pi,1.5260405925833798*pi) q[31];
U1q(0.336274585223673*pi,0.07924302432563124*pi) q[32];
U1q(0.640618470072644*pi,0.5189995969691248*pi) q[33];
U1q(0.114813618637912*pi,0.8337146630585961*pi) q[34];
U1q(1.14323485610237*pi,0.40946293087856356*pi) q[35];
U1q(1.91643602784536*pi,1.7990561992268677*pi) q[36];
U1q(0.661994058698425*pi,1.710959412846103*pi) q[37];
U1q(3.160231282352682*pi,1.395478959037158*pi) q[38];
U1q(1.57382843779697*pi,0.6198064603164979*pi) q[39];
rz(2.8069145509626043*pi) q[0];
rz(2.1584197007535044*pi) q[1];
rz(3.790976476840935*pi) q[2];
rz(1.5385676084277922*pi) q[3];
rz(3.082589691631246*pi) q[4];
rz(2.7648803303984826*pi) q[5];
rz(3.7227325490971417*pi) q[6];
rz(1.5752625138376315*pi) q[7];
rz(1.6746915195209358*pi) q[8];
rz(0.9288461211096997*pi) q[9];
rz(1.6250950759120224*pi) q[10];
rz(3.156494993153898*pi) q[11];
rz(1.7787302852242668*pi) q[12];
rz(1.2043325017378093*pi) q[13];
rz(0.7546611612421454*pi) q[14];
rz(2.147081398384701*pi) q[15];
rz(0.6274113811842508*pi) q[16];
rz(0.6619328804915732*pi) q[17];
rz(3.533981954920773*pi) q[18];
rz(0.41907744028397875*pi) q[19];
rz(0.950963911365398*pi) q[20];
rz(2.6132498694651285*pi) q[21];
rz(3.7228094836349017*pi) q[22];
rz(0.872598268367013*pi) q[23];
rz(1.6812246141085954*pi) q[24];
rz(0.3464241664814711*pi) q[25];
rz(3.134747243519235*pi) q[26];
rz(1.2739269095446923*pi) q[27];
rz(1.1758985252657967*pi) q[28];
rz(0.505553145497788*pi) q[29];
rz(2.535642554844344*pi) q[30];
rz(0.47395940741662024*pi) q[31];
rz(1.9207569756743688*pi) q[32];
rz(3.481000403030875*pi) q[33];
rz(1.166285336941404*pi) q[34];
rz(3.5905370691214364*pi) q[35];
rz(2.2009438007731323*pi) q[36];
rz(0.289040587153897*pi) q[37];
rz(2.604521040962842*pi) q[38];
rz(3.380193539683502*pi) q[39];
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