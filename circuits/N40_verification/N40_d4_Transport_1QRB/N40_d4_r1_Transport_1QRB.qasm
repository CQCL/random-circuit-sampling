OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
rz(0.932782997933917*pi) q[0];
rz(0.372183994660217*pi) q[1];
rz(0.6478005494832031*pi) q[2];
rz(0.643741730242442*pi) q[3];
rz(3.848314064327332*pi) q[4];
rz(2.9341224076522696*pi) q[5];
rz(0.21137153316381124*pi) q[6];
rz(2.39894738229277*pi) q[7];
rz(3.905217400407905*pi) q[8];
rz(0.0271179338270552*pi) q[9];
rz(0.211096264517674*pi) q[10];
rz(0.7992699109017863*pi) q[11];
rz(1.6708815907925325*pi) q[12];
rz(1.22992605632271*pi) q[13];
rz(1.02275957821265*pi) q[14];
rz(0.6170180208574487*pi) q[15];
rz(1.4816900148955061*pi) q[16];
rz(2.59324159453101*pi) q[17];
rz(0.08338530045020198*pi) q[18];
rz(0.729482064470956*pi) q[19];
rz(0.336552429422783*pi) q[20];
rz(3.540279632065457*pi) q[21];
rz(2.251444419830454*pi) q[22];
rz(3.683220838241548*pi) q[23];
rz(0.155375552558211*pi) q[24];
rz(0.09294697550308872*pi) q[25];
rz(2.9674199510339685*pi) q[26];
rz(3.588863872474688*pi) q[27];
rz(0.366933862099862*pi) q[28];
rz(2.24243994648778*pi) q[29];
rz(0.5785979711622876*pi) q[30];
rz(3.606250182664188*pi) q[31];
rz(1.40658287852463*pi) q[32];
rz(0.37090310014270267*pi) q[33];
rz(0.013638179567838415*pi) q[34];
rz(0.602956760939439*pi) q[35];
rz(2.73812306792225*pi) q[36];
rz(0.032294160596474386*pi) q[37];
rz(0.8720776949344962*pi) q[38];
rz(0.6179267152391856*pi) q[39];
U1q(0.460403151528112*pi,0.105572902506872*pi) q[0];
U1q(0.420496925352953*pi,1.4386857022419979*pi) q[1];
U1q(1.39160939863497*pi,0.965211250557484*pi) q[2];
U1q(0.725005515735728*pi,0.405014524962761*pi) q[3];
U1q(0.136076901104*pi,1.408337697994427*pi) q[4];
U1q(1.39232264553851*pi,1.348319476084082*pi) q[5];
U1q(1.33267700087229*pi,1.9554533811738224*pi) q[6];
U1q(0.836920829884951*pi,1.716629831164902*pi) q[7];
U1q(0.522637558933794*pi,1.9547894332059275*pi) q[8];
U1q(0.257281744420108*pi,0.41693855594589*pi) q[9];
U1q(0.25075200851758*pi,1.1850407002493601*pi) q[10];
U1q(1.89697373741509*pi,1.16575847639746*pi) q[11];
U1q(1.28522038548505*pi,1.58414982160119*pi) q[12];
U1q(0.126612151396976*pi,0.301724416342578*pi) q[13];
U1q(0.5532931118125*pi,0.807776187391179*pi) q[14];
U1q(1.90545631231683*pi,1.01751468686431*pi) q[15];
U1q(1.05961047747849*pi,0.875241876480853*pi) q[16];
U1q(0.852727331677412*pi,1.90344437983386*pi) q[17];
U1q(3.393725963723813*pi,1.260331957485337*pi) q[18];
U1q(0.492164691058014*pi,0.963494775345061*pi) q[19];
U1q(0.378238379676752*pi,0.644174991245893*pi) q[20];
U1q(0.851097078595637*pi,0.200328141550611*pi) q[21];
U1q(3.6563417872395982*pi,1.1836040631503129*pi) q[22];
U1q(1.85120560807085*pi,0.95902182009618*pi) q[23];
U1q(0.363412798579925*pi,1.269699870093145*pi) q[24];
U1q(1.59149000514309*pi,1.1410174734029739*pi) q[25];
U1q(3.565419545923281*pi,1.4981896017121419*pi) q[26];
U1q(0.398982086089528*pi,0.618828665798556*pi) q[27];
U1q(0.791435663693186*pi,0.237553534615615*pi) q[28];
U1q(0.694774608045186*pi,1.855005262814035*pi) q[29];
U1q(3.312109864172352*pi,1.642921837414013*pi) q[30];
U1q(0.176585376180332*pi,0.82022559514524*pi) q[31];
U1q(0.549214942591771*pi,0.849557348726525*pi) q[32];
U1q(1.30470233077236*pi,0.521407582101144*pi) q[33];
U1q(1.69827025147264*pi,1.325444308048459*pi) q[34];
U1q(1.53271063041408*pi,1.13868674242285*pi) q[35];
U1q(1.42638951909422*pi,1.588137826803139*pi) q[36];
U1q(3.005587616018161*pi,1.4350841078721661*pi) q[37];
U1q(1.29008731440317*pi,0.591404137127629*pi) q[38];
U1q(1.66816647792793*pi,0.767715501231497*pi) q[39];
RZZ(0.0*pi) q[0],q[17];
RZZ(0.0*pi) q[1],q[36];
RZZ(0.0*pi) q[2],q[34];
RZZ(0.0*pi) q[24],q[3];
RZZ(0.0*pi) q[39],q[4];
RZZ(0.0*pi) q[5],q[16];
RZZ(0.0*pi) q[22],q[6];
RZZ(0.0*pi) q[14],q[7];
RZZ(0.0*pi) q[30],q[8];
RZZ(0.0*pi) q[20],q[9];
RZZ(0.0*pi) q[38],q[10];
RZZ(0.0*pi) q[11],q[28];
RZZ(0.0*pi) q[37],q[12];
RZZ(0.0*pi) q[13],q[25];
RZZ(0.0*pi) q[15],q[29];
RZZ(0.0*pi) q[18],q[31];
RZZ(0.0*pi) q[19],q[21];
RZZ(0.0*pi) q[23],q[35];
RZZ(0.0*pi) q[33],q[26];
RZZ(0.0*pi) q[27],q[32];
rz(3.794830777327903*pi) q[0];
rz(0.126564403356087*pi) q[1];
rz(2.94169438864049*pi) q[2];
rz(0.949325859638435*pi) q[3];
rz(3.369063426067625*pi) q[4];
rz(3.68504659552675*pi) q[5];
rz(1.22453092790916*pi) q[6];
rz(1.00380314766889*pi) q[7];
rz(3.613195839655069*pi) q[8];
rz(0.968630954867747*pi) q[9];
rz(2.3601405464408*pi) q[10];
rz(1.73237455578566*pi) q[11];
rz(0.196938054670278*pi) q[12];
rz(0.960576658266797*pi) q[13];
rz(3.847686558159962*pi) q[14];
rz(0.681730880433212*pi) q[15];
rz(0.3741412979213*pi) q[16];
rz(0.176126899848521*pi) q[17];
rz(3.538442458486238*pi) q[18];
rz(3.639623308919361*pi) q[19];
rz(0.908063343815105*pi) q[20];
rz(3.698161576599899*pi) q[21];
rz(0.147520429750443*pi) q[22];
rz(0.411475462453462*pi) q[23];
rz(0.851907032097803*pi) q[24];
rz(0.366283352253384*pi) q[25];
rz(3.731188658425732*pi) q[26];
rz(0.485391109201429*pi) q[27];
rz(0.399240030021975*pi) q[28];
rz(3.6534697569029992*pi) q[29];
rz(0.237653606107939*pi) q[30];
rz(3.657664494767383*pi) q[31];
rz(0.469698773360787*pi) q[32];
rz(1.22531890838953*pi) q[33];
rz(3.887927109477091*pi) q[34];
rz(0.955598427893491*pi) q[35];
rz(2.4433004444012703*pi) q[36];
rz(3.079154276172445*pi) q[37];
rz(0.747556142628643*pi) q[38];
rz(0.958222663427401*pi) q[39];
U1q(0.2855450665455*pi,1.050758249462798*pi) q[0];
U1q(0.901622483727974*pi,1.820637470370234*pi) q[1];
U1q(0.556707140629717*pi,0.0896498611228488*pi) q[2];
U1q(0.738876130315778*pi,0.861873098210463*pi) q[3];
U1q(0.736417664542269*pi,0.168911872242947*pi) q[4];
U1q(0.169770136029438*pi,0.19202635233291*pi) q[5];
U1q(0.548794350094239*pi,0.435447102024904*pi) q[6];
U1q(0.492104704156263*pi,1.30562259089537*pi) q[7];
U1q(0.271587998804844*pi,0.0934875988533177*pi) q[8];
U1q(0.537514193132415*pi,0.690015187807361*pi) q[9];
U1q(0.686526576071915*pi,1.3255287622284109*pi) q[10];
U1q(0.742126616540874*pi,0.960592521948937*pi) q[11];
U1q(0.437298241307732*pi,0.323912368048714*pi) q[12];
U1q(0.560849474111712*pi,0.212223337408951*pi) q[13];
U1q(0.687817886172246*pi,0.1730564155084*pi) q[14];
U1q(0.676732769545348*pi,0.706140906197589*pi) q[15];
U1q(0.351312489204854*pi,0.20443818186684*pi) q[16];
U1q(0.142836332359293*pi,1.503950257771999*pi) q[17];
U1q(0.300921819588408*pi,1.260770040754218*pi) q[18];
U1q(0.974497208373645*pi,0.200025464753456*pi) q[19];
U1q(0.441836573030636*pi,1.854210471773657*pi) q[20];
U1q(0.476025387243561*pi,1.692032095525011*pi) q[21];
U1q(0.761169550689229*pi,0.790339970730045*pi) q[22];
U1q(0.385562301817522*pi,0.941224932253935*pi) q[23];
U1q(0.115532255702965*pi,1.35474545771132*pi) q[24];
U1q(0.548473239411197*pi,2.70817756914399e-05*pi) q[25];
U1q(0.566405424690225*pi,0.21950251185479*pi) q[26];
U1q(0.138144645810044*pi,1.773915680070548*pi) q[27];
U1q(0.710374117792941*pi,0.760303489475238*pi) q[28];
U1q(0.792307478005541*pi,1.827854505594834*pi) q[29];
U1q(0.487632444515409*pi,0.921962154417124*pi) q[30];
U1q(0.559373467409508*pi,0.470797574781481*pi) q[31];
U1q(0.217674293066612*pi,0.220280303054728*pi) q[32];
U1q(0.570837940747241*pi,0.706315782382453*pi) q[33];
U1q(0.555104706653233*pi,0.309974028096153*pi) q[34];
U1q(0.0956244410100115*pi,1.751447888252386*pi) q[35];
U1q(0.520216039673741*pi,1.7125613492147869*pi) q[36];
U1q(0.746627456337478*pi,1.957696852245985*pi) q[37];
U1q(0.465728014432827*pi,0.342386991155675*pi) q[38];
U1q(0.576715753813256*pi,1.07112831053778*pi) q[39];
RZZ(0.0*pi) q[0],q[35];
RZZ(0.0*pi) q[1],q[17];
RZZ(0.0*pi) q[2],q[27];
RZZ(0.0*pi) q[33],q[3];
RZZ(0.0*pi) q[4],q[24];
RZZ(0.0*pi) q[5],q[8];
RZZ(0.0*pi) q[6],q[20];
RZZ(0.0*pi) q[7],q[13];
RZZ(0.0*pi) q[9],q[21];
RZZ(0.0*pi) q[10],q[32];
RZZ(0.0*pi) q[11],q[34];
RZZ(0.0*pi) q[31],q[12];
RZZ(0.0*pi) q[14],q[22];
RZZ(0.0*pi) q[15],q[38];
RZZ(0.0*pi) q[16],q[25];
RZZ(0.0*pi) q[18],q[19];
RZZ(0.0*pi) q[23],q[37];
RZZ(0.0*pi) q[26],q[36];
RZZ(0.0*pi) q[39],q[28];
RZZ(0.0*pi) q[30],q[29];
rz(0.0200771638984627*pi) q[0];
rz(1.12758541628045*pi) q[1];
rz(2.8867811864394497*pi) q[2];
rz(2.3427952634071403*pi) q[3];
rz(2.63795721771364*pi) q[4];
rz(1.33413274071191*pi) q[5];
rz(0.163388237184131*pi) q[6];
rz(1.13216488141738*pi) q[7];
rz(0.239354855344892*pi) q[8];
rz(3.771535005134744*pi) q[9];
rz(0.483427101974652*pi) q[10];
rz(2.78624260548207*pi) q[11];
rz(0.165689737654971*pi) q[12];
rz(1.2707187652452*pi) q[13];
rz(0.59612274545142*pi) q[14];
rz(2.64728057006018*pi) q[15];
rz(3.534041685725658*pi) q[16];
rz(1.11191956335372*pi) q[17];
rz(1.35435549062115*pi) q[18];
rz(3.035595230054648*pi) q[19];
rz(1.14889042164522*pi) q[20];
rz(1.77887098480037*pi) q[21];
rz(1.70058603606773*pi) q[22];
rz(1.3529854421441*pi) q[23];
rz(0.696943936112095*pi) q[24];
rz(0.141559225174875*pi) q[25];
rz(1.03969863666484*pi) q[26];
rz(2.72023347408599*pi) q[27];
rz(1.47168741166708*pi) q[28];
rz(0.58529006178906*pi) q[29];
rz(0.092027592103924*pi) q[30];
rz(0.507969131213104*pi) q[31];
rz(1.78602713184395*pi) q[32];
rz(1.23161359795208*pi) q[33];
rz(3.150063985784619*pi) q[34];
rz(2.94920014638509*pi) q[35];
rz(3.515607763299564*pi) q[36];
rz(3.748836167802178*pi) q[37];
rz(1.32678962252739*pi) q[38];
rz(3.233567469177769*pi) q[39];
U1q(0.619664754661068*pi,1.797445546245146*pi) q[0];
U1q(0.629078745158977*pi,0.388403048792987*pi) q[1];
U1q(0.578482923875916*pi,1.941000108309468*pi) q[2];
U1q(0.716450608670581*pi,1.848425620247978*pi) q[3];
U1q(0.657329884733118*pi,1.9081817803746899*pi) q[4];
U1q(0.231993497071656*pi,1.752285523032202*pi) q[5];
U1q(0.142621239408808*pi,0.478450450086384*pi) q[6];
U1q(0.520113733482612*pi,1.22790877777281*pi) q[7];
U1q(0.974942922916331*pi,1.891151267632016*pi) q[8];
U1q(0.586279491960754*pi,1.73270381050007*pi) q[9];
U1q(0.103265841143012*pi,1.635979023053833*pi) q[10];
U1q(0.700986883560496*pi,0.0399588335990204*pi) q[11];
U1q(0.524337984600554*pi,0.61790593357269*pi) q[12];
U1q(0.156677172873099*pi,1.2634292902684*pi) q[13];
U1q(0.417350183251868*pi,0.766831559687431*pi) q[14];
U1q(0.51338510003203*pi,1.508980519708391*pi) q[15];
U1q(0.399475912961518*pi,0.93581714986405*pi) q[16];
U1q(0.403036950680605*pi,0.0462777876959947*pi) q[17];
U1q(0.610576016370791*pi,0.677450119566021*pi) q[18];
U1q(0.724491186658224*pi,1.9666796506886648*pi) q[19];
U1q(0.526556721609073*pi,0.481703454927653*pi) q[20];
U1q(0.843816508355697*pi,0.801608249197677*pi) q[21];
U1q(0.777778737678637*pi,1.12462603570181*pi) q[22];
U1q(0.657317436534516*pi,1.18246764062667*pi) q[23];
U1q(0.566278592890563*pi,0.242642189836126*pi) q[24];
U1q(0.598172892797717*pi,1.9768098838342831*pi) q[25];
U1q(0.629036223904551*pi,0.412203489577295*pi) q[26];
U1q(0.620107258680212*pi,1.4590769496055551*pi) q[27];
U1q(0.733659647953143*pi,0.598517960431794*pi) q[28];
U1q(0.255363894165211*pi,0.724838631293754*pi) q[29];
U1q(0.0874380331991998*pi,0.463560246013636*pi) q[30];
U1q(0.177655158183744*pi,0.590251819061177*pi) q[31];
U1q(0.790510461952845*pi,1.43120446356611*pi) q[32];
U1q(0.380704701337491*pi,1.705662408541025*pi) q[33];
U1q(0.824669178055262*pi,1.669329606266151*pi) q[34];
U1q(0.600934102004865*pi,1.675727779843034*pi) q[35];
U1q(0.267473392321651*pi,1.5067681625491751*pi) q[36];
U1q(0.823286639951802*pi,0.0888696301119682*pi) q[37];
U1q(0.39416026433149*pi,1.57733526412941*pi) q[38];
U1q(0.745727355472621*pi,1.51797633362563*pi) q[39];
RZZ(0.0*pi) q[0],q[16];
RZZ(0.0*pi) q[37],q[1];
RZZ(0.0*pi) q[2],q[9];
RZZ(0.0*pi) q[3],q[34];
RZZ(0.0*pi) q[4],q[28];
RZZ(0.0*pi) q[5],q[26];
RZZ(0.0*pi) q[6],q[32];
RZZ(0.0*pi) q[7],q[25];
RZZ(0.0*pi) q[19],q[8];
RZZ(0.0*pi) q[10],q[31];
RZZ(0.0*pi) q[11],q[23];
RZZ(0.0*pi) q[22],q[12];
RZZ(0.0*pi) q[13],q[36];
RZZ(0.0*pi) q[14],q[24];
RZZ(0.0*pi) q[15],q[30];
RZZ(0.0*pi) q[17],q[35];
RZZ(0.0*pi) q[18],q[29];
RZZ(0.0*pi) q[20],q[21];
RZZ(0.0*pi) q[39],q[27];
RZZ(0.0*pi) q[33],q[38];
rz(3.667923283611641*pi) q[0];
rz(1.24172005873701*pi) q[1];
rz(3.276954165428971*pi) q[2];
rz(1.4339904638667*pi) q[3];
rz(2.4142890053644397*pi) q[4];
rz(0.0137505150810817*pi) q[5];
rz(0.956764826607121*pi) q[6];
rz(0.992830235256006*pi) q[7];
rz(3.9502442424432562*pi) q[8];
rz(1.42984676386473*pi) q[9];
rz(0.189477474142504*pi) q[10];
rz(2.0869657300241498*pi) q[11];
rz(0.481180339610989*pi) q[12];
rz(3.818802959784814*pi) q[13];
rz(1.3529116145982*pi) q[14];
rz(0.22033543704846*pi) q[15];
rz(0.688721716920304*pi) q[16];
rz(1.09759560242412*pi) q[17];
rz(1.24253694944423*pi) q[18];
rz(0.087230183743312*pi) q[19];
rz(1.09605227425113*pi) q[20];
rz(2.0743320443549598*pi) q[21];
rz(3.6843438873552428*pi) q[22];
rz(1.48330484291462*pi) q[23];
rz(0.425803743651529*pi) q[24];
rz(0.090096905564618*pi) q[25];
rz(1.04885840088314*pi) q[26];
rz(2.7441781795534803*pi) q[27];
rz(1.01186389702999*pi) q[28];
rz(3.093458911656581*pi) q[29];
rz(2.4991256097937002*pi) q[30];
rz(2.95404977076357*pi) q[31];
rz(0.670624824168*pi) q[32];
rz(1.42043109843714*pi) q[33];
rz(1.40505679046205*pi) q[34];
rz(0.836462993778637*pi) q[35];
rz(3.807558488838382*pi) q[36];
rz(0.560394199348538*pi) q[37];
rz(2.45831416217897*pi) q[38];
rz(3.540886292743768*pi) q[39];
U1q(0.135535721023356*pi,0.76981230048225*pi) q[0];
U1q(0.712704015570973*pi,1.0492629086219*pi) q[1];
U1q(0.554383016990843*pi,0.0994562535428306*pi) q[2];
U1q(0.487196544794441*pi,0.732601528012209*pi) q[3];
U1q(0.632186374595522*pi,1.51366909394549*pi) q[4];
U1q(0.1310036190666*pi,0.414603576667785*pi) q[5];
U1q(0.32702671270251*pi,1.752619812524434*pi) q[6];
U1q(0.8327192622186*pi,0.443691968201115*pi) q[7];
U1q(0.603892189845528*pi,0.144429691884531*pi) q[8];
U1q(0.417244814217624*pi,1.47144969791484*pi) q[9];
U1q(0.390889547540576*pi,0.597318764536665*pi) q[10];
U1q(0.574286977146632*pi,1.346332163735525*pi) q[11];
U1q(0.313322028149841*pi,1.323596513345948*pi) q[12];
U1q(0.538124502086111*pi,1.138173655454829*pi) q[13];
U1q(0.466303271068578*pi,1.37581187546103*pi) q[14];
U1q(0.548996651127588*pi,0.190572944796296*pi) q[15];
U1q(0.30959203168625*pi,1.20783860608266*pi) q[16];
U1q(0.294370582027001*pi,1.632421786600736*pi) q[17];
U1q(0.314414495573805*pi,1.54653536709865*pi) q[18];
U1q(0.808834716518416*pi,1.835108995835163*pi) q[19];
U1q(0.605606239289918*pi,0.775156833197306*pi) q[20];
U1q(0.903106683303421*pi,1.47597223675514*pi) q[21];
U1q(0.45785507485974*pi,1.4082582343605599*pi) q[22];
U1q(0.583375227052349*pi,1.42213656505662*pi) q[23];
U1q(0.769699870496412*pi,0.404780541235391*pi) q[24];
U1q(0.299653446492953*pi,0.891337358731232*pi) q[25];
U1q(0.462835625447772*pi,1.1508569238164*pi) q[26];
U1q(0.663073297224694*pi,1.539980816838251*pi) q[27];
U1q(0.698752527695372*pi,0.834303361729008*pi) q[28];
U1q(0.647287621199599*pi,1.536814808864056*pi) q[29];
U1q(0.615058780877771*pi,1.23574515552307*pi) q[30];
U1q(0.752141623419924*pi,1.9477866306441658*pi) q[31];
U1q(0.717886984628383*pi,0.232648896244315*pi) q[32];
U1q(0.489433911687233*pi,0.541366854772933*pi) q[33];
U1q(0.226154846359704*pi,1.20180287441359*pi) q[34];
U1q(0.47506063420545*pi,0.934006249673411*pi) q[35];
U1q(0.184176157830128*pi,0.482269211365518*pi) q[36];
U1q(0.414648208808111*pi,1.574956457207609*pi) q[37];
U1q(0.603659926545483*pi,1.1227780700893502*pi) q[38];
U1q(0.185382755581731*pi,0.423900236747054*pi) q[39];
RZZ(0.0*pi) q[0],q[7];
RZZ(0.0*pi) q[18],q[1];
RZZ(0.0*pi) q[2],q[39];
RZZ(0.0*pi) q[14],q[3];
RZZ(0.0*pi) q[4],q[34];
RZZ(0.0*pi) q[5],q[37];
RZZ(0.0*pi) q[6],q[13];
RZZ(0.0*pi) q[10],q[8];
RZZ(0.0*pi) q[9],q[28];
RZZ(0.0*pi) q[11],q[12];
RZZ(0.0*pi) q[15],q[33];
RZZ(0.0*pi) q[19],q[16];
RZZ(0.0*pi) q[24],q[17];
RZZ(0.0*pi) q[20],q[25];
RZZ(0.0*pi) q[38],q[21];
RZZ(0.0*pi) q[22],q[23];
RZZ(0.0*pi) q[27],q[26];
RZZ(0.0*pi) q[31],q[29];
RZZ(0.0*pi) q[30],q[32];
RZZ(0.0*pi) q[35],q[36];
rz(3.1935400090073562*pi) q[0];
rz(3.374700119763587*pi) q[1];
rz(1.98089297756212*pi) q[2];
rz(0.779956178729717*pi) q[3];
rz(1.51838175551896*pi) q[4];
rz(1.11584048207344*pi) q[5];
rz(3.6721876579554*pi) q[6];
rz(1.12571441732515*pi) q[7];
rz(0.12312553793797987*pi) q[8];
rz(3.684345049106913*pi) q[9];
rz(3.407685280613323*pi) q[10];
rz(1.45179813070521*pi) q[11];
rz(2.67859493125191*pi) q[12];
rz(1.40649656173752*pi) q[13];
rz(0.26555650983422*pi) q[14];
rz(1.3604457432988*pi) q[15];
rz(0.949083251771693*pi) q[16];
rz(0.882858500473073*pi) q[17];
rz(2.6140844362077402*pi) q[18];
rz(1.60846841788099*pi) q[19];
rz(2.53548406972916*pi) q[20];
rz(2.10190096626641*pi) q[21];
rz(2.52211750185993*pi) q[22];
rz(0.280261753623651*pi) q[23];
rz(2.79489448527184*pi) q[24];
rz(0.290938779400344*pi) q[25];
rz(3.294740933444972*pi) q[26];
rz(2.06653197925582*pi) q[27];
rz(2.17811876871953*pi) q[28];
rz(3.762853405466116*pi) q[29];
rz(0.208281760113797*pi) q[30];
rz(3.23786125424517*pi) q[31];
rz(3.264510906334713*pi) q[32];
rz(1.02514683970428*pi) q[33];
rz(0.0114244200697671*pi) q[34];
rz(2.01241973994177*pi) q[35];
rz(2.91555007040219*pi) q[36];
rz(2.62542141261151*pi) q[37];
rz(1.0617414044608*pi) q[38];
rz(1.13472941862186*pi) q[39];
U1q(3.267599325470858*pi,0.8009620410001801*pi) q[0];
U1q(3.460794656059455*pi,1.625111236928706*pi) q[1];
U1q(3.109641613633577*pi,1.9009145529541085*pi) q[2];
U1q(3.509742431243112*pi,0.59087242358955*pi) q[3];
U1q(3.331433242263054*pi,1.06191416595445*pi) q[4];
U1q(3.622590119995181*pi,1.28059247546115*pi) q[5];
U1q(3.279762846862157*pi,1.426321212073592*pi) q[6];
U1q(3.508585266975718*pi,0.347331448166318*pi) q[7];
U1q(3.78122138166255*pi,0.6822754038507901*pi) q[8];
U1q(3.424701400871466*pi,1.029006296186946*pi) q[9];
U1q(3.212780643400224*pi,1.127283888284373*pi) q[10];
U1q(3.850135861206451*pi,0.74015755234622*pi) q[11];
U1q(3.57603125049194*pi,1.34277601537384*pi) q[12];
U1q(3.175417991535593*pi,1.24040488160157*pi) q[13];
U1q(3.757766261964527*pi,1.1906725825357*pi) q[14];
U1q(3.338626010131462*pi,1.23487776009056*pi) q[15];
U1q(3.768979146423728*pi,0.42333836484834*pi) q[16];
U1q(3.674257729057061*pi,1.128499986876226*pi) q[17];
U1q(3.167015295193441*pi,1.9241993042057812*pi) q[18];
U1q(3.734705643716123*pi,0.861252454646106*pi) q[19];
U1q(3.532678642271358*pi,0.56942979034876*pi) q[20];
U1q(3.483886835476555*pi,1.796582486175754*pi) q[21];
U1q(3.463437568367743*pi,0.78514987672061*pi) q[22];
U1q(3.40296797911058*pi,0.466444750063035*pi) q[23];
U1q(3.549058210423363*pi,1.680038871294468*pi) q[24];
U1q(3.579433656587324*pi,1.838472248558863*pi) q[25];
U1q(3.320819274399606*pi,1.079279919291587*pi) q[26];
U1q(3.951625721278069*pi,1.401523211522983*pi) q[27];
U1q(3.645073531435189*pi,0.895330553779266*pi) q[28];
U1q(3.522114240361287*pi,1.20619825316329*pi) q[29];
U1q(3.258156826201259*pi,1.575858433383649*pi) q[30];
U1q(3.520295005815962*pi,0.070453595398003*pi) q[31];
U1q(3.755951458585551*pi,0.8640273283464901*pi) q[32];
U1q(3.709929493780767*pi,0.21498478492797*pi) q[33];
U1q(3.077982766093868*pi,1.4839980165775462*pi) q[34];
U1q(3.885737260787583*pi,0.847574741391151*pi) q[35];
U1q(3.287460666867776*pi,0.5106487084207401*pi) q[36];
U1q(3.403757381124091*pi,0.50657517115109*pi) q[37];
U1q(3.9461461935561015*pi,0.411742093215479*pi) q[38];
U1q(3.32880752011437*pi,1.9273186277582375*pi) q[39];
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
