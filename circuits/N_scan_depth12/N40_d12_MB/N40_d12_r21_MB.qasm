OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(1.11148596287506*pi,0.06637316156863719*pi) q[0];
U1q(1.37689198148472*pi,0.11633090828471289*pi) q[1];
U1q(0.461656967567412*pi,1.137532150938989*pi) q[2];
U1q(1.70879204269982*pi,0.18452419201211792*pi) q[3];
U1q(0.650776364740162*pi,1.2115041521374361*pi) q[4];
U1q(0.576116663211938*pi,0.0214067343641553*pi) q[5];
U1q(0.276523192701583*pi,1.286683993239138*pi) q[6];
U1q(3.644715033244741*pi,0.5609555790119306*pi) q[7];
U1q(1.80474017095264*pi,1.6145001109841834*pi) q[8];
U1q(0.789242760103662*pi,1.4120573619891799*pi) q[9];
U1q(3.703959430853525*pi,0.5673916352169055*pi) q[10];
U1q(0.490525536924152*pi,0.282401128015284*pi) q[11];
U1q(0.34906865143261*pi,1.272387537605753*pi) q[12];
U1q(1.42481409033659*pi,0.04489923908699144*pi) q[13];
U1q(0.382441523585493*pi,0.6374360518270701*pi) q[14];
U1q(1.89736994865879*pi,1.7928771467455757*pi) q[15];
U1q(3.634269628143453*pi,0.9711526611815707*pi) q[16];
U1q(1.92707578713013*pi,0.7200816564914602*pi) q[17];
U1q(0.466740449637112*pi,0.112902668684177*pi) q[18];
U1q(0.699434468354454*pi,0.742561113621234*pi) q[19];
U1q(1.66223424674337*pi,0.48403184412676986*pi) q[20];
U1q(0.632028770312364*pi,1.011241166348169*pi) q[21];
U1q(0.261708225009557*pi,1.581366666020072*pi) q[22];
U1q(1.28446437497113*pi,0.3695291158889065*pi) q[23];
U1q(0.555245579066911*pi,1.53551683515981*pi) q[24];
U1q(1.47835098579306*pi,1.972751584557747*pi) q[25];
U1q(0.745155877516078*pi,0.183926295263402*pi) q[26];
U1q(1.83929388451269*pi,1.2691135527792476*pi) q[27];
U1q(1.62920840195712*pi,1.9541621959192796*pi) q[28];
U1q(1.47991719289172*pi,1.8814487159309687*pi) q[29];
U1q(1.32630466154691*pi,0.1888030743411085*pi) q[30];
U1q(1.22007202044365*pi,1.4926353310412614*pi) q[31];
U1q(0.24806085668945*pi,1.9853136772088218*pi) q[32];
U1q(1.39336263630677*pi,1.99948517907522*pi) q[33];
U1q(1.61746019116852*pi,0.041903945249966615*pi) q[34];
U1q(1.66448350477748*pi,1.75942770183082*pi) q[35];
U1q(0.477371580311397*pi,1.774188375447916*pi) q[36];
U1q(0.681325511672942*pi,1.359689321868959*pi) q[37];
U1q(0.564893508243217*pi,1.020462795846457*pi) q[38];
U1q(0.119957800191997*pi,1.898846916646056*pi) q[39];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[1],q[22];
RZZ(0.5*pi) q[12],q[2];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[4],q[35];
RZZ(0.5*pi) q[32],q[5];
RZZ(0.5*pi) q[38],q[7];
RZZ(0.5*pi) q[21],q[8];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[37],q[27];
RZZ(0.5*pi) q[39],q[29];
U1q(0.545316741580981*pi,0.09424538438925723*pi) q[0];
U1q(0.987433485350791*pi,0.40442846397198284*pi) q[1];
U1q(0.366415498692731*pi,0.30568826173206*pi) q[2];
U1q(0.801250390468118*pi,1.762065174437708*pi) q[3];
U1q(0.221366924166864*pi,0.6948109641096498*pi) q[4];
U1q(0.241133172135407*pi,1.79557396295592*pi) q[5];
U1q(0.584404713764741*pi,1.27676862412999*pi) q[6];
U1q(0.131554002455692*pi,1.3875479965390305*pi) q[7];
U1q(0.652031536241425*pi,1.8403387854326634*pi) q[8];
U1q(0.795617881382831*pi,0.09535187216489005*pi) q[9];
U1q(0.653046284556113*pi,0.5337817747052553*pi) q[10];
U1q(0.140781171116695*pi,0.5256698480501498*pi) q[11];
U1q(0.503678131885248*pi,1.2982882043890398*pi) q[12];
U1q(0.12619998949915*pi,0.8029861184553417*pi) q[13];
U1q(0.221860051357769*pi,1.4179085635690898*pi) q[14];
U1q(0.568797595055647*pi,1.6500506963181554*pi) q[15];
U1q(0.224202247544294*pi,0.1366921663921108*pi) q[16];
U1q(0.18967262839317*pi,1.0876918019332802*pi) q[17];
U1q(0.508962021240977*pi,1.48105573826841*pi) q[18];
U1q(0.260329485230697*pi,0.9155105696976*pi) q[19];
U1q(0.347051696312624*pi,0.57824001353541*pi) q[20];
U1q(0.327836072885753*pi,0.5280932987278*pi) q[21];
U1q(0.330966673746887*pi,1.43893729198669*pi) q[22];
U1q(0.778136783660672*pi,1.7012188470255363*pi) q[23];
U1q(0.735274528425982*pi,1.7468103811275801*pi) q[24];
U1q(0.201749155315042*pi,0.631209268122207*pi) q[25];
U1q(0.70114189125294*pi,0.37185911581657005*pi) q[26];
U1q(0.79007959897437*pi,1.5570509815622176*pi) q[27];
U1q(0.0667804559392678*pi,0.7939951675461496*pi) q[28];
U1q(0.199507225513095*pi,1.655728158025899*pi) q[29];
U1q(0.64416280522926*pi,1.7181539826031984*pi) q[30];
U1q(0.827578341709779*pi,1.9371807634293816*pi) q[31];
U1q(0.821520622586832*pi,1.4790314724580602*pi) q[32];
U1q(0.570615435805043*pi,0.43173196151187*pi) q[33];
U1q(0.414024191735501*pi,0.31887860538450674*pi) q[34];
U1q(0.71583924469366*pi,1.7992234953897501*pi) q[35];
U1q(0.349643962367491*pi,1.4149606724157402*pi) q[36];
U1q(0.150565808225826*pi,1.1111325895908002*pi) q[37];
U1q(0.750691891023839*pi,1.1239838530398796*pi) q[38];
U1q(0.245313271978522*pi,0.6089726168976799*pi) q[39];
RZZ(0.5*pi) q[29],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[3],q[16];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[20],q[7];
RZZ(0.5*pi) q[8],q[19];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[17],q[35];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[21],q[22];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[38],q[37];
U1q(0.664676393183429*pi,0.7483988972677573*pi) q[0];
U1q(0.15601486009722*pi,0.07918798744880284*pi) q[1];
U1q(0.370582147680049*pi,1.77464452740766*pi) q[2];
U1q(0.676345268710086*pi,0.8975609355521379*pi) q[3];
U1q(0.26117456499264*pi,1.0655270694048404*pi) q[4];
U1q(0.500468715469279*pi,0.17205259319195987*pi) q[5];
U1q(0.317112622946773*pi,1.87738403978212*pi) q[6];
U1q(0.462488834255139*pi,0.48857463736599094*pi) q[7];
U1q(0.31779336728461*pi,0.5431000069482632*pi) q[8];
U1q(0.636127449823921*pi,0.6154742447860198*pi) q[9];
U1q(0.727503400536291*pi,0.3220244846851852*pi) q[10];
U1q(0.725216771097301*pi,0.0417981579935498*pi) q[11];
U1q(0.791259675811038*pi,1.2097020785911496*pi) q[12];
U1q(0.826493443470273*pi,1.5569474310509408*pi) q[13];
U1q(0.82361292672394*pi,0.4547398895836503*pi) q[14];
U1q(0.780142568744731*pi,1.3341335915706658*pi) q[15];
U1q(0.540743210065592*pi,1.3092018867113109*pi) q[16];
U1q(0.242865875347003*pi,0.6415767778502701*pi) q[17];
U1q(0.853489046548744*pi,0.09780297256458992*pi) q[18];
U1q(0.813383965144678*pi,1.12092302797193*pi) q[19];
U1q(0.568687728681496*pi,0.870517505694*pi) q[20];
U1q(0.11197300901009*pi,0.2817333884305002*pi) q[21];
U1q(0.116188721229377*pi,1.1631318830598696*pi) q[22];
U1q(0.634550124010935*pi,1.8435430214794364*pi) q[23];
U1q(0.425204013955789*pi,0.3013071564797398*pi) q[24];
U1q(0.434828500673254*pi,1.403360336571387*pi) q[25];
U1q(0.31551016610033*pi,1.0232491301282796*pi) q[26];
U1q(0.519290175093971*pi,0.9567047991072277*pi) q[27];
U1q(0.390272480243125*pi,0.30318517093857*pi) q[28];
U1q(0.369840734812969*pi,0.8402518159076084*pi) q[29];
U1q(0.565661463020013*pi,0.6774543258457886*pi) q[30];
U1q(0.767800246892806*pi,0.18846457633547153*pi) q[31];
U1q(0.330235242131265*pi,0.34199071826876004*pi) q[32];
U1q(0.535376873725817*pi,1.6263420089872*pi) q[33];
U1q(0.683834517092658*pi,1.9304043860572966*pi) q[34];
U1q(0.832402789412487*pi,0.55799851774439*pi) q[35];
U1q(0.65272905395299*pi,1.4925019988996002*pi) q[36];
U1q(0.427631509773637*pi,0.07114538238968038*pi) q[37];
U1q(0.196798009804708*pi,0.7086383330694392*pi) q[38];
U1q(0.0743897578896717*pi,1.5041047111372299*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[30],q[4];
RZZ(0.5*pi) q[25],q[5];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[32],q[8];
RZZ(0.5*pi) q[9],q[35];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[12],q[24];
RZZ(0.5*pi) q[39],q[15];
RZZ(0.5*pi) q[26],q[16];
RZZ(0.5*pi) q[17],q[19];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[22],q[23];
RZZ(0.5*pi) q[29],q[36];
RZZ(0.5*pi) q[31],q[33];
U1q(0.46007793415195*pi,1.1187857403152073*pi) q[0];
U1q(0.338100836788129*pi,1.0724166010430132*pi) q[1];
U1q(0.482041936467716*pi,0.29613785496236034*pi) q[2];
U1q(0.432495865051158*pi,0.03445050286926765*pi) q[3];
U1q(0.743647392926035*pi,0.8726715492968697*pi) q[4];
U1q(0.116488473328093*pi,1.6850871151197993*pi) q[5];
U1q(0.781142841910001*pi,1.0450296196897906*pi) q[6];
U1q(0.23260144000918*pi,1.1154094585096104*pi) q[7];
U1q(0.360562150848737*pi,1.1484236310271436*pi) q[8];
U1q(0.588318211175459*pi,0.1449768220913601*pi) q[9];
U1q(0.490785169925874*pi,0.3882536279810056*pi) q[10];
U1q(0.652229672350893*pi,0.6770324669437997*pi) q[11];
U1q(0.413081540249431*pi,0.75728674308436*pi) q[12];
U1q(0.348683138421831*pi,0.23417624284770078*pi) q[13];
U1q(0.814022278360789*pi,0.007259618067159579*pi) q[14];
U1q(0.332212761868893*pi,1.094378599232586*pi) q[15];
U1q(0.11420886640351*pi,1.0960658326841513*pi) q[16];
U1q(0.410961789024341*pi,0.19649645486026124*pi) q[17];
U1q(0.257173511750658*pi,0.8606345105124698*pi) q[18];
U1q(0.0678791238971028*pi,1.2494983999749802*pi) q[19];
U1q(0.901931507543077*pi,0.08535558076917038*pi) q[20];
U1q(0.51836670547895*pi,0.002536431514659654*pi) q[21];
U1q(0.218635228619911*pi,1.5622684690096698*pi) q[22];
U1q(0.264380364923411*pi,1.7932730698078565*pi) q[23];
U1q(0.555129429721649*pi,0.6833615867537404*pi) q[24];
U1q(0.733226697226946*pi,0.1348485348950268*pi) q[25];
U1q(0.772312166697896*pi,0.26711770480496*pi) q[26];
U1q(0.649355296002367*pi,0.6969313543249971*pi) q[27];
U1q(0.349723481978054*pi,0.5985100357403095*pi) q[28];
U1q(0.839582301801207*pi,0.3303492940496584*pi) q[29];
U1q(0.620940258691237*pi,1.5597552176601184*pi) q[30];
U1q(0.21898262647231*pi,0.02776114548419173*pi) q[31];
U1q(0.216971948384509*pi,0.30930335680668986*pi) q[32];
U1q(0.584331351604257*pi,0.46963202258515047*pi) q[33];
U1q(0.803123737935995*pi,0.30858217467555704*pi) q[34];
U1q(0.673737969330586*pi,1.0613750117830794*pi) q[35];
U1q(0.554487474957311*pi,1.9925903574561996*pi) q[36];
U1q(0.22860853325747*pi,0.5265799104260704*pi) q[37];
U1q(0.609262926682254*pi,1.6379244321335005*pi) q[38];
U1q(0.0601153098716926*pi,0.20666374005455967*pi) q[39];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[8],q[6];
RZZ(0.5*pi) q[9],q[7];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[36];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[37],q[24];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[30],q[32];
RZZ(0.5*pi) q[39],q[31];
RZZ(0.5*pi) q[38],q[35];
U1q(0.534358794189115*pi,0.49894390690815715*pi) q[0];
U1q(0.479121894044404*pi,0.8784105703393621*pi) q[1];
U1q(0.94565155175222*pi,1.0253152919984991*pi) q[2];
U1q(0.291175959354316*pi,1.8904894712258073*pi) q[3];
U1q(0.441974062100072*pi,0.44613632093162003*pi) q[4];
U1q(0.491670360375662*pi,1.6613423386604005*pi) q[5];
U1q(0.15792536441406*pi,1.2417661061921006*pi) q[6];
U1q(0.356373022368441*pi,1.8844327619658205*pi) q[7];
U1q(0.549578446934367*pi,0.27506541936989315*pi) q[8];
U1q(0.309620381173281*pi,0.20338849146214955*pi) q[9];
U1q(0.438052249652687*pi,1.726348943212546*pi) q[10];
U1q(0.640777512256393*pi,0.2970888204659996*pi) q[11];
U1q(0.637182200913289*pi,0.06850181533758004*pi) q[12];
U1q(0.462765175932645*pi,1.2177478232988914*pi) q[13];
U1q(0.381160259724954*pi,0.3758979864493206*pi) q[14];
U1q(0.893921767893723*pi,0.729526926521677*pi) q[15];
U1q(0.439563580331016*pi,0.9942680842582607*pi) q[16];
U1q(0.45497274401816*pi,0.574708269565761*pi) q[17];
U1q(0.494507664423367*pi,0.9674162988973105*pi) q[18];
U1q(0.60037308103104*pi,0.33556185327247956*pi) q[19];
U1q(0.686571177434008*pi,0.66798070930838*pi) q[20];
U1q(0.517472656689236*pi,1.6380222886914204*pi) q[21];
U1q(0.201424957906863*pi,0.5450895859058296*pi) q[22];
U1q(0.185082299345506*pi,1.1322901484081065*pi) q[23];
U1q(0.470146726663948*pi,1.8013506225720004*pi) q[24];
U1q(0.344576707574404*pi,1.1185612500322861*pi) q[25];
U1q(0.584011541200964*pi,0.15281127922352056*pi) q[26];
U1q(0.602604567549883*pi,0.6654650815480867*pi) q[27];
U1q(0.526677969089539*pi,0.40446405336787983*pi) q[28];
U1q(0.653431756496056*pi,1.2713385929855683*pi) q[29];
U1q(0.602638593029429*pi,0.2353395863140486*pi) q[30];
U1q(0.639724884169332*pi,1.2133832548105623*pi) q[31];
U1q(0.6966472343039*pi,0.2814252357629403*pi) q[32];
U1q(0.728219440850133*pi,1.3908661501388497*pi) q[33];
U1q(0.528469955892235*pi,0.3855649009582667*pi) q[34];
U1q(0.378286062030305*pi,1.2392216873785404*pi) q[35];
U1q(0.273445742292199*pi,1.6369694292993007*pi) q[36];
U1q(0.221439161187847*pi,1.7080160375745006*pi) q[37];
U1q(0.188554542577462*pi,1.3480167678479997*pi) q[38];
U1q(0.705484855753945*pi,1.6075336135710998*pi) q[39];
RZZ(0.5*pi) q[0],q[8];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[3],q[38];
RZZ(0.5*pi) q[34],q[4];
RZZ(0.5*pi) q[33],q[5];
RZZ(0.5*pi) q[23],q[6];
RZZ(0.5*pi) q[14],q[7];
RZZ(0.5*pi) q[18],q[9];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[24];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[31],q[22];
RZZ(0.5*pi) q[28],q[36];
RZZ(0.5*pi) q[39],q[32];
U1q(0.572889919653918*pi,1.201274390662638*pi) q[0];
U1q(0.454297842138445*pi,0.8009019955115129*pi) q[1];
U1q(0.470646108355886*pi,0.5800197125123994*pi) q[2];
U1q(0.687675379773486*pi,0.2710259641405379*pi) q[3];
U1q(0.798870119537406*pi,1.5886733158375002*pi) q[4];
U1q(0.593436913065031*pi,0.7943898463539991*pi) q[5];
U1q(0.773876689203642*pi,1.7435607929400003*pi) q[6];
U1q(0.640687672631485*pi,0.289416872198629*pi) q[7];
U1q(0.322194822661285*pi,1.6789440106774833*pi) q[8];
U1q(0.677306604890659*pi,1.5884367770204992*pi) q[9];
U1q(0.760385290038526*pi,1.596096759259705*pi) q[10];
U1q(0.40427342097825*pi,0.32026297014243*pi) q[11];
U1q(0.364093952466388*pi,0.6164319017004001*pi) q[12];
U1q(0.625090686002936*pi,1.8787291793376912*pi) q[13];
U1q(0.696715269451676*pi,1.5104268573271007*pi) q[14];
U1q(0.402938647574868*pi,0.8633144273064772*pi) q[15];
U1q(0.841018689819556*pi,0.5057920783563699*pi) q[16];
U1q(0.885002092909064*pi,0.40897067184016045*pi) q[17];
U1q(0.313137322999944*pi,0.5590690677707997*pi) q[18];
U1q(0.646362682607852*pi,0.9951644859752804*pi) q[19];
U1q(0.159919530497249*pi,1.418033136925871*pi) q[20];
U1q(0.86915024436092*pi,1.9446368554621003*pi) q[21];
U1q(0.593534846235303*pi,0.6716376943405002*pi) q[22];
U1q(0.616966469311256*pi,1.681130586273408*pi) q[23];
U1q(0.365106785039527*pi,1.4961868771874993*pi) q[24];
U1q(0.926902933311325*pi,0.740812782079816*pi) q[25];
U1q(0.336180922059736*pi,0.0205805838360007*pi) q[26];
U1q(0.542124285358197*pi,1.866152725582948*pi) q[27];
U1q(0.191478773255432*pi,0.3913344559204788*pi) q[28];
U1q(0.668194383106742*pi,1.551068519476468*pi) q[29];
U1q(0.192241823841295*pi,1.2476686496640195*pi) q[30];
U1q(0.487456040147743*pi,0.4606286253423626*pi) q[31];
U1q(0.354218438691319*pi,1.7876538750329*pi) q[32];
U1q(0.703912642121525*pi,1.1582989633736211*pi) q[33];
U1q(0.61334939611985*pi,0.36012282129696693*pi) q[34];
U1q(0.575514273853016*pi,0.42788826501345056*pi) q[35];
U1q(0.681316452569279*pi,1.6565113172725*pi) q[36];
U1q(0.570726637663516*pi,1.6773047460236992*pi) q[37];
U1q(0.978983924119408*pi,1.9175692813759007*pi) q[38];
U1q(0.816730066092886*pi,1.1241212790255997*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[33];
RZZ(0.5*pi) q[37],q[2];
RZZ(0.5*pi) q[39],q[3];
RZZ(0.5*pi) q[12],q[4];
RZZ(0.5*pi) q[36],q[5];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[21],q[7];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[11],q[16];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[32],q[24];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[34],q[29];
U1q(0.505814767641667*pi,0.7780573834334383*pi) q[0];
U1q(0.627731405023503*pi,1.943106591211512*pi) q[1];
U1q(0.203554727559691*pi,1.8095721132585005*pi) q[2];
U1q(0.482572596597949*pi,0.01420121898861737*pi) q[3];
U1q(0.191774465352402*pi,1.2136668661994996*pi) q[4];
U1q(0.515620314779256*pi,1.6825693231051986*pi) q[5];
U1q(0.615743939779462*pi,0.9671886405498995*pi) q[6];
U1q(0.651035013729521*pi,1.7603195036700292*pi) q[7];
U1q(0.841576164627615*pi,0.688519307175083*pi) q[8];
U1q(0.484033332696072*pi,1.7107389854201003*pi) q[9];
U1q(0.396015069188102*pi,0.859539220706905*pi) q[10];
U1q(0.738197487019635*pi,0.7638785978352001*pi) q[11];
U1q(0.794598229474828*pi,0.49669912533840055*pi) q[12];
U1q(0.323060942749926*pi,0.8390773747149911*pi) q[13];
U1q(0.396969289538225*pi,0.12596835123519945*pi) q[14];
U1q(0.882937393207874*pi,0.336115868587477*pi) q[15];
U1q(0.283828139092383*pi,0.42322526533227034*pi) q[16];
U1q(0.862691634806506*pi,0.14561509071496026*pi) q[17];
U1q(0.395699900390466*pi,1.1910552002266002*pi) q[18];
U1q(0.515559725174513*pi,0.6775091198749603*pi) q[19];
U1q(0.364340392412313*pi,1.9415271718264702*pi) q[20];
U1q(0.537324010497355*pi,0.5978783116129005*pi) q[21];
U1q(0.718003853682932*pi,1.7211197248710999*pi) q[22];
U1q(0.30490647544998*pi,1.7239311807625057*pi) q[23];
U1q(0.727737797541374*pi,1.9985347806183995*pi) q[24];
U1q(0.806109669620183*pi,0.6882436343077476*pi) q[25];
U1q(0.0488794931425639*pi,0.6531713422387995*pi) q[26];
U1q(0.487792989062265*pi,1.8730642312826475*pi) q[27];
U1q(0.851056311660564*pi,0.03383447239457915*pi) q[28];
U1q(0.653169629459265*pi,0.7250021772136677*pi) q[29];
U1q(0.849836318518362*pi,0.38706222657400957*pi) q[30];
U1q(0.28779055793676*pi,1.9942495597493597*pi) q[31];
U1q(0.621225789583005*pi,1.8204957597934008*pi) q[32];
U1q(0.747202507878191*pi,1.0285077947807206*pi) q[33];
U1q(0.619196242302422*pi,0.8451860615623676*pi) q[34];
U1q(0.558496454511976*pi,0.5774555679577205*pi) q[35];
U1q(0.367330211929859*pi,0.5186808867660009*pi) q[36];
U1q(0.770495837750399*pi,1.6669418935200007*pi) q[37];
U1q(0.713170210562972*pi,1.5673180043213009*pi) q[38];
U1q(0.233925826025184*pi,1.0051258001470007*pi) q[39];
rz(2.1119927896602615*pi) q[0];
rz(3.469116221452488*pi) q[1];
rz(1.027040218873701*pi) q[2];
rz(1.300814733749382*pi) q[3];
rz(0.26030434622450116*pi) q[4];
rz(0.09510269166619878*pi) q[5];
rz(1.8536043741044992*pi) q[6];
rz(2.1755759161736705*pi) q[7];
rz(0.6154587941945167*pi) q[8];
rz(3.3588666719069007*pi) q[9];
rz(0.0673878268076944*pi) q[10];
rz(1.7314645275966*pi) q[11];
rz(3.9701782824334*pi) q[12];
rz(3.1273839272586095*pi) q[13];
rz(1.4309040503618995*pi) q[14];
rz(0.9918564809870247*pi) q[15];
rz(0.8970104307338307*pi) q[16];
rz(3.838444029216939*pi) q[17];
rz(3.301684992104599*pi) q[18];
rz(1.1065977213232898*pi) q[19];
rz(0.7996636440306304*pi) q[20];
rz(2.1941518568191007*pi) q[21];
rz(0.7777823912000006*pi) q[22];
rz(1.4901002939213939*pi) q[23];
rz(1.3137718874781008*pi) q[24];
rz(3.936976072623054*pi) q[25];
rz(1.9469357534341007*pi) q[26];
rz(2.363179178833253*pi) q[27];
rz(0.7467977721844186*pi) q[28];
rz(1.095532317958531*pi) q[29];
rz(1.92108388263029*pi) q[30];
rz(0.03500412908223893*pi) q[31];
rz(3.5683543414655006*pi) q[32];
rz(1.7263392249693794*pi) q[33];
rz(0.4759550824674328*pi) q[34];
rz(0.8269427065029795*pi) q[35];
rz(0.5271308238247983*pi) q[36];
rz(2.9282943101369003*pi) q[37];
rz(1.8634920079579*pi) q[38];
rz(2.263590099559501*pi) q[39];
U1q(1.50581476764167*pi,1.890050173093652*pi) q[0];
U1q(0.627731405023503*pi,0.412222812663957*pi) q[1];
U1q(1.20355472755969*pi,1.836612332132132*pi) q[2];
U1q(1.48257259659795*pi,0.315015952738041*pi) q[3];
U1q(0.191774465352402*pi,0.4739712124239801*pi) q[4];
U1q(0.515620314779256*pi,0.7776720147714*pi) q[5];
U1q(1.61574393977946*pi,1.82079301465445*pi) q[6];
U1q(0.651035013729521*pi,0.935895419843669*pi) q[7];
U1q(1.84157616462762*pi,0.303978101369574*pi) q[8];
U1q(1.48403333269607*pi,0.0696056573270631*pi) q[9];
U1q(1.3960150691881*pi,1.9269270475145603*pi) q[10];
U1q(1.73819748701964*pi,1.49534312543177*pi) q[11];
U1q(1.79459822947483*pi,1.4668774077718179*pi) q[12];
U1q(1.32306094274993*pi,0.9664613019736801*pi) q[13];
U1q(3.396969289538225*pi,0.556872401597122*pi) q[14];
U1q(0.882937393207874*pi,0.327972349574494*pi) q[15];
U1q(0.283828139092383*pi,0.320235696066077*pi) q[16];
U1q(0.862691634806506*pi,0.984059119931954*pi) q[17];
U1q(1.39569990039047*pi,1.492740192331198*pi) q[18];
U1q(0.515559725174513*pi,0.784106841198255*pi) q[19];
U1q(0.364340392412313*pi,1.741190815857091*pi) q[20];
U1q(1.53732401049736*pi,1.792030168432041*pi) q[21];
U1q(0.718003853682932*pi,1.4989021160711231*pi) q[22];
U1q(0.30490647544998*pi,0.214031474683838*pi) q[23];
U1q(1.72773779754137*pi,0.312306668096436*pi) q[24];
U1q(1.80610966962018*pi,1.625219706930789*pi) q[25];
U1q(0.0488794931425639*pi,1.600107095672855*pi) q[26];
U1q(0.487792989062265*pi,1.2362434101158661*pi) q[27];
U1q(0.851056311660564*pi,1.780632244579008*pi) q[28];
U1q(0.653169629459265*pi,0.820534495172183*pi) q[29];
U1q(1.84983631851836*pi,1.30814610920434*pi) q[30];
U1q(3.28779055793676*pi,1.029253688831604*pi) q[31];
U1q(0.621225789583005*pi,0.388850101258975*pi) q[32];
U1q(0.747202507878191*pi,1.754847019750115*pi) q[33];
U1q(0.619196242302422*pi,0.321141144029795*pi) q[34];
U1q(0.558496454511976*pi,0.404398274460638*pi) q[35];
U1q(1.36733021192986*pi,0.0458117105908332*pi) q[36];
U1q(1.7704958377504*pi,1.595236203656861*pi) q[37];
U1q(1.71317021056297*pi,0.430810012279254*pi) q[38];
U1q(1.23392582602518*pi,0.268715899706487*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[33];
RZZ(0.5*pi) q[37],q[2];
RZZ(0.5*pi) q[39],q[3];
RZZ(0.5*pi) q[12],q[4];
RZZ(0.5*pi) q[36],q[5];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[21],q[7];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[11],q[16];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[32],q[24];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[34],q[29];
U1q(1.57288991965392*pi,1.4668331658643685*pi) q[0];
U1q(0.454297842138445*pi,0.270018216963977*pi) q[1];
U1q(1.47064610835589*pi,0.06616473287818514*pi) q[2];
U1q(3.312324620226513*pi,0.0581912075861509*pi) q[3];
U1q(0.798870119537406*pi,0.84897766206204*pi) q[4];
U1q(0.593436913065031*pi,1.8894925380201801*pi) q[5];
U1q(3.226123310796358*pi,0.04442086226438613*pi) q[6];
U1q(0.640687672631485*pi,1.46499278837226*pi) q[7];
U1q(1.32219482266128*pi,0.31355339786713665*pi) q[8];
U1q(3.32269339510934*pi,0.19190786572670937*pi) q[9];
U1q(3.760385290038527*pi,0.1903695089617381*pi) q[10];
U1q(1.40427342097825*pi,1.9389587531244983*pi) q[11];
U1q(3.364093952466388*pi,1.3471446314097928*pi) q[12];
U1q(1.62509068600294*pi,1.9268094973510421*pi) q[13];
U1q(1.69671526945168*pi,0.17241389550528963*pi) q[14];
U1q(0.402938647574868*pi,1.8551709082934602*pi) q[15];
U1q(0.841018689819556*pi,0.40280250909019*pi) q[16];
U1q(1.88500209290906*pi,1.24741470105717*pi) q[17];
U1q(1.31313732299994*pi,1.124726324787074*pi) q[18];
U1q(1.64636268260785*pi,0.101762207298575*pi) q[19];
U1q(1.15991953049725*pi,0.21769678095645006*pi) q[20];
U1q(1.86915024436092*pi,0.44527162458284986*pi) q[21];
U1q(1.5935348462353*pi,0.44942008554048996*pi) q[22];
U1q(0.616966469311256*pi,0.171230880194786*pi) q[23];
U1q(1.36510678503953*pi,0.8146545715273199*pi) q[24];
U1q(3.073097066688676*pi,1.5726505591587334*pi) q[25];
U1q(0.336180922059736*pi,1.967516337270064*pi) q[26];
U1q(1.5421242853582*pi,1.229331904416152*pi) q[27];
U1q(1.19147877325543*pi,1.138132228104876*pi) q[28];
U1q(1.66819438310674*pi,0.646600837434905*pi) q[29];
U1q(3.807758176158704*pi,1.4475396861143355*pi) q[30];
U1q(3.512543959852257*pi,0.5628746232386062*pi) q[31];
U1q(1.35421843869132*pi,0.3560082164984899*pi) q[32];
U1q(1.70391264212152*pi,0.8846381883430701*pi) q[33];
U1q(1.61334939611985*pi,0.83607790376433*pi) q[34];
U1q(0.575514273853016*pi,1.254830971516413*pi) q[35];
U1q(1.68131645256928*pi,0.9079812800843733*pi) q[36];
U1q(3.570726637663516*pi,0.584873351153115*pi) q[37];
U1q(3.02101607588059*pi,0.08055873522468751*pi) q[38];
U1q(3.183269933907114*pi,0.14972042082787929*pi) q[39];
RZZ(0.5*pi) q[0],q[8];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[3],q[38];
RZZ(0.5*pi) q[34],q[4];
RZZ(0.5*pi) q[33],q[5];
RZZ(0.5*pi) q[23],q[6];
RZZ(0.5*pi) q[14],q[7];
RZZ(0.5*pi) q[18],q[9];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[24];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[31],q[22];
RZZ(0.5*pi) q[28],q[36];
RZZ(0.5*pi) q[39],q[32];
U1q(3.534358794189115*pi,1.7645026821098464*pi) q[0];
U1q(0.479121894044404*pi,1.34752679179183*pi) q[1];
U1q(0.94565155175222*pi,1.5114603123642751*pi) q[2];
U1q(1.29117595935432*pi,0.4387277005008834*pi) q[3];
U1q(1.44197406210007*pi,1.70644066715613*pi) q[4];
U1q(0.491670360375662*pi,0.7564450303266299*pi) q[5];
U1q(1.15792536441406*pi,1.5462155490122977*pi) q[6];
U1q(0.356373022368441*pi,0.06000867813946997*pi) q[7];
U1q(0.549578446934367*pi,1.9096748065595026*pi) q[8];
U1q(1.30962038117328*pi,0.5769561512850309*pi) q[9];
U1q(0.438052249652687*pi,0.3206216929145682*pi) q[10];
U1q(1.64077751225639*pi,0.9157846034480683*pi) q[11];
U1q(1.63718220091329*pi,0.7992145450469368*pi) q[12];
U1q(0.462765175932645*pi,1.2658281413122623*pi) q[13];
U1q(0.381160259724954*pi,0.037885024627557495*pi) q[14];
U1q(0.893921767893723*pi,1.7213834075086298*pi) q[15];
U1q(0.439563580331016*pi,1.891278514992038*pi) q[16];
U1q(1.45497274401816*pi,1.0816771033316304*pi) q[17];
U1q(1.49450766442337*pi,1.5330735559136142*pi) q[18];
U1q(3.39962691896896*pi,1.7613648400013737*pi) q[19];
U1q(1.68657117743401*pi,1.9677492085739314*pi) q[20];
U1q(0.517472656689236*pi,1.1386570578121469*pi) q[21];
U1q(3.798575042093136*pi,0.575968193975144*pi) q[22];
U1q(0.185082299345506*pi,1.622390442329444*pi) q[23];
U1q(0.470146726663948*pi,0.11981831691180411*pi) q[24];
U1q(3.655423292425596*pi,1.194902091206263*pi) q[25];
U1q(3.584011541200964*pi,1.0997470326576009*pi) q[26];
U1q(3.397395432450117*pi,1.4300195484509792*pi) q[27];
U1q(3.473322030910461*pi,0.12500263065746098*pi) q[28];
U1q(3.3465682435039428*pi,0.9263307639257143*pi) q[29];
U1q(3.397361406970571*pi,0.45986874946430545*pi) q[30];
U1q(3.360275115830668*pi,0.810119993770436*pi) q[31];
U1q(1.6966472343039*pi,0.8622368557684972*pi) q[32];
U1q(1.72821944085013*pi,0.6520710015778766*pi) q[33];
U1q(1.52846995589224*pi,0.8106358241030314*pi) q[34];
U1q(1.3782860620303*pi,0.06616439388149997*pi) q[35];
U1q(1.2734457422922*pi,0.8884393921112401*pi) q[36];
U1q(0.221439161187847*pi,1.6155846427039147*pi) q[37];
U1q(1.18855454257746*pi,0.6501112487525849*pi) q[38];
U1q(3.294515144246056*pi,0.6663080862823854*pi) q[39];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[8],q[6];
RZZ(0.5*pi) q[9],q[7];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[36];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[37],q[24];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[30],q[32];
RZZ(0.5*pi) q[39],q[31];
RZZ(0.5*pi) q[38],q[35];
U1q(1.46007793415195*pi,0.14466084870280183*pi) q[0];
U1q(0.338100836788129*pi,0.5415328224954798*pi) q[1];
U1q(0.482041936467716*pi,1.7822828753281348*pi) q[2];
U1q(1.43249586505116*pi,1.5826887321443412*pi) q[3];
U1q(3.256352607073965*pi,0.2799054387908859*pi) q[4];
U1q(1.11648847332809*pi,1.78018980678599*pi) q[5];
U1q(1.78114284191*pi,0.34947906251001365*pi) q[6];
U1q(0.23260144000918*pi,0.2909853746832698*pi) q[7];
U1q(1.36056215084874*pi,1.7830330182167522*pi) q[8];
U1q(0.588318211175459*pi,1.5185444819142369*pi) q[9];
U1q(1.49078516992587*pi,1.9825263776830284*pi) q[10];
U1q(3.347770327649107*pi,1.5358409569702758*pi) q[11];
U1q(1.41308154024943*pi,1.1104296173001629*pi) q[12];
U1q(1.34868313842183*pi,0.2822565608610619*pi) q[13];
U1q(0.814022278360789*pi,1.6692466562453978*pi) q[14];
U1q(0.332212761868893*pi,0.08623508021956994*pi) q[15];
U1q(0.11420886640351*pi,0.99307626341793*pi) q[16];
U1q(0.410961789024341*pi,1.70346528862615*pi) q[17];
U1q(3.742826488249342*pi,0.6398553442984485*pi) q[18];
U1q(3.9321208761028963*pi,1.847428293298878*pi) q[19];
U1q(0.901931507543077*pi,1.3851240800347218*pi) q[20];
U1q(0.51836670547895*pi,0.503171200635389*pi) q[21];
U1q(1.21863522861991*pi,0.5587893108713029*pi) q[22];
U1q(0.264380364923411*pi,0.28337336372921995*pi) q[23];
U1q(0.555129429721649*pi,0.0018292810935738313*pi) q[24];
U1q(3.266773302773054*pi,1.1786148063435231*pi) q[25];
U1q(1.7723121666979*pi,0.9854406070761581*pi) q[26];
U1q(1.64935529600237*pi,0.3985532756740675*pi) q[27];
U1q(3.650276518021946*pi,0.9309566482850199*pi) q[28];
U1q(1.83958230180121*pi,1.8673200628616629*pi) q[29];
U1q(1.62094025869124*pi,0.13545311811823435*pi) q[30];
U1q(3.21898262647231*pi,1.995742103096787*pi) q[31];
U1q(0.216971948384509*pi,0.8901149768122476*pi) q[32];
U1q(0.584331351604257*pi,1.7308368740241766*pi) q[33];
U1q(0.803123737935995*pi,0.7336530978203717*pi) q[34];
U1q(3.326262030669413*pi,1.2440110694769473*pi) q[35];
U1q(3.4455125250426892*pi,1.5328184639543703*pi) q[36];
U1q(0.22860853325747*pi,1.4341485155554556*pi) q[37];
U1q(0.609262926682254*pi,0.940018913038053*pi) q[38];
U1q(3.060115309871692*pi,0.06717795979891084*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[30],q[4];
RZZ(0.5*pi) q[25],q[5];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[32],q[8];
RZZ(0.5*pi) q[9],q[35];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[12],q[24];
RZZ(0.5*pi) q[39],q[15];
RZZ(0.5*pi) q[26],q[16];
RZZ(0.5*pi) q[17],q[19];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[22],q[23];
RZZ(0.5*pi) q[29],q[36];
RZZ(0.5*pi) q[31],q[33];
U1q(1.66467639318343*pi,0.7742740056553612*pi) q[0];
U1q(1.15601486009722*pi,1.5483042089012802*pi) q[1];
U1q(1.37058214768005*pi,1.2607895477734345*pi) q[2];
U1q(3.676345268710086*pi,1.7195782994614721*pi) q[3];
U1q(1.26117456499264*pi,1.0870499186829092*pi) q[4];
U1q(1.50046871546928*pi,0.2932243287138361*pi) q[5];
U1q(1.31711262294677*pi,1.5171246424176823*pi) q[6];
U1q(1.46248883425514*pi,1.6641505535396401*pi) q[7];
U1q(3.682206632715391*pi,1.3883566422956246*pi) q[8];
U1q(0.636127449823921*pi,1.9890419046088939*pi) q[9];
U1q(3.272496599463709*pi,1.0487555209788493*pi) q[10];
U1q(1.7252167710973*pi,1.1710752659205237*pi) q[11];
U1q(0.791259675811038*pi,0.5628449528069508*pi) q[12];
U1q(3.173506556529728*pi,0.9594853726578307*pi) q[13];
U1q(0.82361292672394*pi,1.1167269277618779*pi) q[14];
U1q(0.780142568744731*pi,1.3259900725576603*pi) q[15];
U1q(1.54074321006559*pi,0.2062123174450896*pi) q[16];
U1q(0.242865875347003*pi,0.1485456116161603*pi) q[17];
U1q(3.853489046548744*pi,0.4026868822463339*pi) q[18];
U1q(1.81338396514468*pi,0.9760036653019286*pi) q[19];
U1q(0.568687728681496*pi,1.1702860049595518*pi) q[20];
U1q(1.11197300901009*pi,1.7823681575512387*pi) q[21];
U1q(0.116188721229377*pi,1.1596527249215125*pi) q[22];
U1q(0.634550124010935*pi,1.3336433154008*pi) q[23];
U1q(1.42520401395579*pi,0.6197748508195842*pi) q[24];
U1q(3.565171499326746*pi,0.9101030046671639*pi) q[25];
U1q(0.31551016610033*pi,0.7415720323994781*pi) q[26];
U1q(0.519290175093971*pi,0.6583267204562873*pi) q[27];
U1q(3.609727519756874*pi,1.226281513086767*pi) q[28];
U1q(1.36984073481297*pi,0.3772225847196129*pi) q[29];
U1q(0.565661463020013*pi,1.2531522263039148*pi) q[30];
U1q(3.767800246892806*pi,1.1564455339480766*pi) q[31];
U1q(0.330235242131265*pi,0.9228023382743178*pi) q[32];
U1q(1.53537687372582*pi,1.8875468604262267*pi) q[33];
U1q(1.68383451709266*pi,0.35547530920210146*pi) q[34];
U1q(3.167597210587514*pi,1.7473875635156375*pi) q[35];
U1q(3.34727094604701*pi,0.03290682251096033*pi) q[36];
U1q(0.427631509773637*pi,1.9787139875190656*pi) q[37];
U1q(0.196798009804708*pi,1.010732813974008*pi) q[38];
U1q(0.0743897578896717*pi,0.3646189308815819*pi) q[39];
RZZ(0.5*pi) q[29],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[3],q[16];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[20],q[7];
RZZ(0.5*pi) q[8],q[19];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[17],q[35];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[21],q[22];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[38],q[37];
U1q(3.454683258419019*pi,1.428427518533864*pi) q[0];
U1q(3.012566514649207*pi,1.223063732378094*pi) q[1];
U1q(3.633584501307269*pi,0.7297458134490302*pi) q[2];
U1q(0.801250390468118*pi,0.584082538347042*pi) q[3];
U1q(0.221366924166864*pi,0.7163338133877293*pi) q[4];
U1q(0.241133172135407*pi,0.9167456984777953*pi) q[5];
U1q(1.58440471376474*pi,1.9165092267655561*pi) q[6];
U1q(1.13155400245569*pi,0.765177194366597*pi) q[7];
U1q(3.347968463758575*pi,0.09111786381122489*pi) q[8];
U1q(1.79561788138283*pi,1.4689195319877668*pi) q[9];
U1q(3.346953715443887*pi,1.8369982309587796*pi) q[10];
U1q(0.140781171116695*pi,1.654946955977124*pi) q[11];
U1q(1.50367813188525*pi,0.6514310786048512*pi) q[12];
U1q(3.873800010500849*pi,1.7134466852534302*pi) q[13];
U1q(0.221860051357769*pi,0.07989560174731736*pi) q[14];
U1q(1.56879759505565*pi,1.64190717730515*pi) q[15];
U1q(1.22420224754429*pi,1.3787220377642875*pi) q[16];
U1q(0.18967262839317*pi,0.5946606356991704*pi) q[17];
U1q(0.508962021240977*pi,0.7859396479501519*pi) q[18];
U1q(0.260329485230697*pi,0.7705912070276089*pi) q[19];
U1q(0.347051696312624*pi,0.878008512800962*pi) q[20];
U1q(3.672163927114247*pi,0.5360082472539478*pi) q[21];
U1q(0.330966673746887*pi,1.435458133848333*pi) q[22];
U1q(0.778136783660672*pi,0.19131914094689995*pi) q[23];
U1q(3.264725471574018*pi,0.17427162617175407*pi) q[24];
U1q(1.20174915531504*pi,0.6822540731163915*pi) q[25];
U1q(0.70114189125294*pi,1.090182018087768*pi) q[26];
U1q(1.79007959897437*pi,0.25867290291128775*pi) q[27];
U1q(3.933219544060732*pi,0.735471516479187*pi) q[28];
U1q(3.199507225513094*pi,0.561746242601328*pi) q[29];
U1q(1.64416280522926*pi,0.2938518830613144*pi) q[30];
U1q(3.827578341709779*pi,1.4077293468541736*pi) q[31];
U1q(0.821520622586832*pi,1.059843092463618*pi) q[32];
U1q(3.429384564194957*pi,0.08215690790155517*pi) q[33];
U1q(3.5859758082644992*pi,1.9670010898748878*pi) q[34];
U1q(3.28416075530634*pi,0.5061625858702872*pi) q[35];
U1q(3.650356037632509*pi,0.11044814899482036*pi) q[36];
U1q(0.150565808225826*pi,0.018701194720175174*pi) q[37];
U1q(0.750691891023839*pi,0.42607833394444805*pi) q[38];
U1q(1.24531327197852*pi,0.4694868366420337*pi) q[39];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[1],q[22];
RZZ(0.5*pi) q[12],q[2];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[4],q[35];
RZZ(0.5*pi) q[32],q[5];
RZZ(0.5*pi) q[38],q[7];
RZZ(0.5*pi) q[21],q[8];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[37],q[27];
RZZ(0.5*pi) q[39],q[29];
U1q(1.11148596287506*pi,1.4562997413544752*pi) q[0];
U1q(1.37689198148472*pi,0.5111612880653684*pi) q[1];
U1q(3.461656967567412*pi,1.8979019242421105*pi) q[2];
U1q(0.708792042699816*pi,0.0065415559214521135*pi) q[3];
U1q(0.650776364740162*pi,1.2330270014155094*pi) q[4];
U1q(0.576116663211938*pi,0.14257846988604683*pi) q[5];
U1q(3.2765231927015828*pi,0.9065938576564117*pi) q[6];
U1q(0.644715033244741*pi,0.9385847768394973*pi) q[7];
U1q(1.80474017095264*pi,0.31695653825970815*pi) q[8];
U1q(1.78924276010366*pi,1.1522140421634748*pi) q[9];
U1q(3.703959430853525*pi,0.8033883704471285*pi) q[10];
U1q(0.490525536924152*pi,1.4116782359422633*pi) q[11];
U1q(1.34906865143261*pi,1.6773317453881402*pi) q[12];
U1q(1.42481409033659*pi,1.4715335646217698*pi) q[13];
U1q(0.382441523585493*pi,1.2994230900053072*pi) q[14];
U1q(1.89736994865879*pi,1.4990807268777315*pi) q[15];
U1q(0.634269628143453*pi,0.21318253255374753*pi) q[16];
U1q(0.927075787130133*pi,0.22705049025735047*pi) q[17];
U1q(0.466740449637112*pi,0.41778657836592226*pi) q[18];
U1q(0.699434468354454*pi,0.5976417509512384*pi) q[19];
U1q(0.662234246743374*pi,0.783800343392322*pi) q[20];
U1q(1.63202877031236*pi,0.05286037963356671*pi) q[21];
U1q(0.261708225009557*pi,1.577887507881714*pi) q[22];
U1q(0.284464374971128*pi,0.8596294098102701*pi) q[23];
U1q(1.55524557906691*pi,1.3855651721395192*pi) q[24];
U1q(0.478350985793056*pi,1.023796389551892*pi) q[25];
U1q(0.745155877516078*pi,1.9022491975345979*pi) q[26];
U1q(1.83929388451269*pi,0.5466103316942581*pi) q[27];
U1q(1.62920840195712*pi,0.5753044881060485*pi) q[28];
U1q(0.479917192891721*pi,0.7874668005063974*pi) q[29];
U1q(1.32630466154691*pi,1.8232027913234012*pi) q[30];
U1q(0.220072020443647*pi,0.9631839144660539*pi) q[31];
U1q(0.24806085668945*pi,0.5661252972143878*pi) q[32];
U1q(1.39336263630677*pi,1.5144036903382112*pi) q[33];
U1q(1.61746019116852*pi,0.24397575000943483*pi) q[34];
U1q(1.66448350477748*pi,1.5459583794292158*pi) q[35];
U1q(1.4773715803114*pi,0.7512204459626461*pi) q[36];
U1q(0.681325511672942*pi,0.2672579269983357*pi) q[37];
U1q(0.564893508243217*pi,1.3225572767510183*pi) q[38];
U1q(1.119957800192*pi,1.1796125368936603*pi) q[39];
rz(0.5437002586455248*pi) q[0];
rz(1.4888387119346316*pi) q[1];
rz(2.1020980757578895*pi) q[2];
rz(3.9934584440785477*pi) q[3];
rz(0.7669729985844908*pi) q[4];
rz(3.857421530113953*pi) q[5];
rz(3.0934061423435883*pi) q[6];
rz(1.0614152231605027*pi) q[7];
rz(3.683043461740292*pi) q[8];
rz(0.8477859578365252*pi) q[9];
rz(1.1966116295528715*pi) q[10];
rz(2.5883217640577367*pi) q[11];
rz(2.3226682546118598*pi) q[12];
rz(2.5284664353782302*pi) q[13];
rz(0.7005769099946928*pi) q[14];
rz(2.5009192731222685*pi) q[15];
rz(3.7868174674462525*pi) q[16];
rz(1.7729495097426495*pi) q[17];
rz(3.5822134216340777*pi) q[18];
rz(3.4023582490487616*pi) q[19];
rz(3.216199656607678*pi) q[20];
rz(1.9471396203664333*pi) q[21];
rz(0.42211249211828594*pi) q[22];
rz(3.14037059018973*pi) q[23];
rz(0.6144348278604808*pi) q[24];
rz(2.976203610448108*pi) q[25];
rz(2.097750802465402*pi) q[26];
rz(3.453389668305742*pi) q[27];
rz(1.4246955118939515*pi) q[28];
rz(3.2125331994936026*pi) q[29];
rz(2.1767972086765988*pi) q[30];
rz(3.036816085533946*pi) q[31];
rz(3.4338747027856122*pi) q[32];
rz(2.485596309661789*pi) q[33];
rz(1.7560242499905652*pi) q[34];
rz(0.45404162057078423*pi) q[35];
rz(1.2487795540373539*pi) q[36];
rz(3.7327420730016643*pi) q[37];
rz(2.6774427232489817*pi) q[38];
rz(2.8203874631063397*pi) q[39];
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