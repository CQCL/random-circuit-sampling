OPENQASM 2.0;
include "hqslib1.inc";

qreg q[16];
creg c[16];
U1q(1.64989045789324*pi,1.3496993806163426*pi) q[0];
U1q(0.612208831044778*pi,0.221601067298764*pi) q[1];
U1q(0.665425715418614*pi,1.588203330147947*pi) q[2];
U1q(1.50854262641651*pi,1.5601958618348104*pi) q[3];
U1q(0.333888152913213*pi,1.7208442291834949*pi) q[4];
U1q(1.57955992244662*pi,0.33268384794955635*pi) q[5];
U1q(1.58207957229967*pi,1.972948224252217*pi) q[6];
U1q(0.615155496016884*pi,0.559635115911066*pi) q[7];
U1q(0.456666849989641*pi,1.825269952457677*pi) q[8];
U1q(3.5579013367059478*pi,1.4026742665379504*pi) q[9];
U1q(1.94294151372963*pi,1.3223048776210167*pi) q[10];
U1q(0.717329392522986*pi,0.204984153786796*pi) q[11];
U1q(0.305689800173609*pi,0.747211978064722*pi) q[12];
U1q(1.38287493340427*pi,0.5786880897850202*pi) q[13];
U1q(3.494195371451959*pi,1.300656232463483*pi) q[14];
U1q(0.582770636967392*pi,0.946210239441139*pi) q[15];
RZZ(0.5*pi) q[5],q[0];
RZZ(0.5*pi) q[13],q[1];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[4],q[14];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[11],q[12];
U1q(0.444856588752441*pi,1.2359976220048328*pi) q[0];
U1q(0.961489416242804*pi,1.6943551781977702*pi) q[1];
U1q(0.65857317207042*pi,1.7252686054016402*pi) q[2];
U1q(0.540710511748001*pi,0.3429926315758105*pi) q[3];
U1q(0.501295677179257*pi,0.6352500089698101*pi) q[4];
U1q(0.542069659710466*pi,0.5297032379402165*pi) q[5];
U1q(0.681424301497946*pi,0.820620758424087*pi) q[6];
U1q(0.935715256189045*pi,0.8863436591716001*pi) q[7];
U1q(0.34783916916543*pi,1.70860428152867*pi) q[8];
U1q(0.640874806081873*pi,1.5609246379060302*pi) q[9];
U1q(0.44752648725383*pi,1.4162838111424265*pi) q[10];
U1q(0.327878177835848*pi,0.8618501090651902*pi) q[11];
U1q(0.127831042876454*pi,0.12290966293963002*pi) q[12];
U1q(0.662732126023775*pi,0.2501062230287603*pi) q[13];
U1q(0.649789240014749*pi,0.5928746076049589*pi) q[14];
U1q(0.266481048109189*pi,0.5796477388579899*pi) q[15];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[15],q[14];
U1q(0.562727598406556*pi,1.7124783523470324*pi) q[0];
U1q(0.408977838913434*pi,0.20576666063283966*pi) q[1];
U1q(0.305702633302232*pi,1.8386656266108998*pi) q[2];
U1q(0.418555441051815*pi,1.2742100866071002*pi) q[3];
U1q(0.435578796836121*pi,0.16669023436850017*pi) q[4];
U1q(0.403765087133137*pi,0.7125221779347166*pi) q[5];
U1q(0.414755737407428*pi,0.9291939069521571*pi) q[6];
U1q(0.327305168839848*pi,1.26344581279304*pi) q[7];
U1q(0.618791627344742*pi,1.7010626549184398*pi) q[8];
U1q(0.693924690841137*pi,0.4622606429162204*pi) q[9];
U1q(0.342975509942436*pi,1.6900925280474368*pi) q[10];
U1q(0.238667409872714*pi,1.63873401771591*pi) q[11];
U1q(0.365700488534244*pi,1.7675514336280598*pi) q[12];
U1q(0.361302770462001*pi,0.026049863137109774*pi) q[13];
U1q(0.70210838288164*pi,1.1658764477351626*pi) q[14];
U1q(0.68567236925422*pi,1.1063477534210602*pi) q[15];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[7],q[13];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[14],q[12];
U1q(0.196719206117285*pi,0.2909749201150724*pi) q[0];
U1q(0.776871128861356*pi,1.4388379152477402*pi) q[1];
U1q(0.373281302059252*pi,1.9221168352445002*pi) q[2];
U1q(0.594034882594698*pi,1.6922658408228104*pi) q[3];
U1q(0.689498495561887*pi,0.7693302773878701*pi) q[4];
U1q(0.477445360153671*pi,1.6754044468767262*pi) q[5];
U1q(0.841110587332439*pi,1.388655568576847*pi) q[6];
U1q(0.751228052802804*pi,1.5784839909017796*pi) q[7];
U1q(0.115172375021241*pi,1.8482487264646004*pi) q[8];
U1q(0.308772669515241*pi,1.2993666443450502*pi) q[9];
U1q(0.186279634407053*pi,1.9962862446052156*pi) q[10];
U1q(0.267812747305795*pi,0.12747493896223006*pi) q[11];
U1q(0.0878303588669823*pi,0.12003710221541031*pi) q[12];
U1q(0.807844281405938*pi,1.0475276182838096*pi) q[13];
U1q(0.119608142708833*pi,0.2162533786448133*pi) q[14];
U1q(0.292168418499218*pi,0.8950666031494308*pi) q[15];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[6],q[2];
RZZ(0.5*pi) q[9],q[3];
RZZ(0.5*pi) q[4],q[11];
RZZ(0.5*pi) q[5],q[13];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[15],q[10];
U1q(0.463776923451998*pi,1.2021211090101236*pi) q[0];
U1q(0.138101890947642*pi,0.032069444301069616*pi) q[1];
U1q(0.684929655317789*pi,0.7838508128212798*pi) q[2];
U1q(0.561668018690558*pi,0.6847022261959204*pi) q[3];
U1q(0.790559769765364*pi,1.69504053391028*pi) q[4];
U1q(0.577049891442843*pi,0.841123917880326*pi) q[5];
U1q(0.480108556050109*pi,1.6679352445134565*pi) q[6];
U1q(0.210714488272742*pi,1.4858197520522598*pi) q[7];
U1q(0.800009947484342*pi,0.46009764832879974*pi) q[8];
U1q(0.641083480748062*pi,1.41613621570884*pi) q[9];
U1q(0.762097917850383*pi,1.1856501453562558*pi) q[10];
U1q(0.134551150929843*pi,1.8739099159237007*pi) q[11];
U1q(0.764609285333162*pi,0.3831044648190396*pi) q[12];
U1q(0.741522747036552*pi,1.52312121049472*pi) q[13];
U1q(0.673173748834378*pi,0.8877220909425532*pi) q[14];
U1q(0.785192380114459*pi,0.54151618666579*pi) q[15];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[5],q[8];
RZZ(0.5*pi) q[7],q[11];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[10],q[14];
RZZ(0.5*pi) q[13],q[15];
U1q(0.571289009596447*pi,0.7360868760442436*pi) q[0];
U1q(0.752628322153345*pi,0.5414361371144896*pi) q[1];
U1q(0.555707122871275*pi,0.5598886550454996*pi) q[2];
U1q(0.735945094495029*pi,0.7432485025906104*pi) q[3];
U1q(0.321365702944776*pi,0.5901789752927504*pi) q[4];
U1q(0.714699774895072*pi,0.7331727606529075*pi) q[5];
U1q(0.470357017717316*pi,0.17063679123951658*pi) q[6];
U1q(0.248762713262672*pi,0.6477982080659004*pi) q[7];
U1q(0.297975199021678*pi,0.7967469585593996*pi) q[8];
U1q(0.722026942724276*pi,0.6850908359155206*pi) q[9];
U1q(0.626943829087542*pi,1.330956831713216*pi) q[10];
U1q(0.732784170416063*pi,1.8626490358380998*pi) q[11];
U1q(0.397362708181691*pi,1.6128167907250006*pi) q[12];
U1q(0.659364972041555*pi,1.6027234056620205*pi) q[13];
U1q(0.687349076875115*pi,0.21626398534848335*pi) q[14];
U1q(0.327976409276544*pi,1.6904757438123994*pi) q[15];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[8],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[4],q[15];
RZZ(0.5*pi) q[5],q[7];
RZZ(0.5*pi) q[13],q[6];
RZZ(0.5*pi) q[9],q[11];
U1q(0.351120324028246*pi,0.5079780555000433*pi) q[0];
U1q(0.887805806334919*pi,1.7342115738617991*pi) q[1];
U1q(0.0971152350588255*pi,1.7180678769921993*pi) q[2];
U1q(0.814180009640704*pi,0.5592241740408106*pi) q[3];
U1q(0.177152010123753*pi,1.33202036519066*pi) q[4];
U1q(0.423507083397649*pi,0.9363870034482566*pi) q[5];
U1q(0.185253871623309*pi,1.9526445574584166*pi) q[6];
U1q(0.48481551715452*pi,1.7591752297643009*pi) q[7];
U1q(0.161744896895609*pi,0.07731031263989863*pi) q[8];
U1q(0.685903929868883*pi,1.6161097436132401*pi) q[9];
U1q(0.614639718976052*pi,0.808133770751617*pi) q[10];
U1q(0.361777748290385*pi,1.6232596610098007*pi) q[11];
U1q(0.549800356838746*pi,1.2570765234797001*pi) q[12];
U1q(0.812549429387169*pi,0.8840255418518197*pi) q[13];
U1q(0.769752608057195*pi,0.5523578716008828*pi) q[14];
U1q(0.444572784793199*pi,0.6333360969720996*pi) q[15];
rz(2.8722002389383565*pi) q[0];
rz(1.1273709292532992*pi) q[1];
rz(2.3613254991463997*pi) q[2];
rz(1.1270617273774892*pi) q[3];
rz(1.34383567112112*pi) q[4];
rz(3.5975872324340425*pi) q[5];
rz(0.8994964291187824*pi) q[6];
rz(3.5894106380413007*pi) q[7];
rz(1.5365339202970993*pi) q[8];
rz(3.66576996311647*pi) q[9];
rz(2.463558112414603*pi) q[10];
rz(1.8656523417068005*pi) q[11];
rz(3.1512009426337997*pi) q[12];
rz(3.85822323658328*pi) q[13];
rz(2.6397390252780166*pi) q[14];
rz(2.2467153597734004*pi) q[15];
U1q(0.351120324028246*pi,0.380178294438444*pi) q[0];
U1q(0.887805806334919*pi,1.861582503115177*pi) q[1];
U1q(1.09711523505883*pi,1.079393376138586*pi) q[2];
U1q(0.814180009640704*pi,0.686285901418296*pi) q[3];
U1q(1.17715201012375*pi,1.675856036311785*pi) q[4];
U1q(0.423507083397649*pi,1.533974235882343*pi) q[5];
U1q(1.18525387162331*pi,1.852140986577191*pi) q[6];
U1q(1.48481551715452*pi,0.348585867805668*pi) q[7];
U1q(0.161744896895609*pi,0.613844232936973*pi) q[8];
U1q(3.685903929868883*pi,0.281879706729714*pi) q[9];
U1q(1.61463971897605*pi,0.271691883166217*pi) q[10];
U1q(1.36177774829039*pi,0.488912002716509*pi) q[11];
U1q(1.54980035683875*pi,1.40827746611352*pi) q[12];
U1q(1.81254942938717*pi,1.7422487784350609*pi) q[13];
U1q(0.769752608057195*pi,0.19209689687888*pi) q[14];
U1q(1.4445727847932*pi,1.880051456745506*pi) q[15];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[8],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[4],q[15];
RZZ(0.5*pi) q[5],q[7];
RZZ(0.5*pi) q[13],q[6];
RZZ(0.5*pi) q[9],q[11];
U1q(0.571289009596447*pi,1.60828711498266*pi) q[0];
U1q(0.752628322153345*pi,1.668807066367838*pi) q[1];
U1q(3.444292877128725*pi,0.2375725980852681*pi) q[2];
U1q(3.7359450944950288*pi,0.870310229968043*pi) q[3];
U1q(3.321365702944776*pi,1.4176974262097066*pi) q[4];
U1q(1.71469977489507*pi,1.3307599930869598*pi) q[5];
U1q(3.529642982282684*pi,1.634148752796119*pi) q[6];
U1q(3.751237286737328*pi,0.4599628895041139*pi) q[7];
U1q(0.297975199021678*pi,0.3332808788564099*pi) q[8];
U1q(3.277973057275724*pi,1.212898614427443*pi) q[9];
U1q(3.373056170912458*pi,1.7488688222046038*pi) q[10];
U1q(3.267215829583937*pi,0.2495226278881878*pi) q[11];
U1q(3.602637291818309*pi,0.05253719886829272*pi) q[12];
U1q(3.340635027958446*pi,1.0235509146248098*pi) q[13];
U1q(0.687349076875115*pi,0.85600301062646*pi) q[14];
U1q(3.672023590723456*pi,0.8229118099052157*pi) q[15];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[5],q[8];
RZZ(0.5*pi) q[7],q[11];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[10],q[14];
RZZ(0.5*pi) q[13],q[15];
U1q(0.463776923451998*pi,0.07432134794850986*pi) q[0];
U1q(0.138101890947642*pi,1.1594403735544199*pi) q[1];
U1q(1.68492965531779*pi,1.0136104403094968*pi) q[2];
U1q(3.438331981309442*pi,1.9288565063627228*pi) q[3];
U1q(1.79055976976536*pi,0.5225589848272367*pi) q[4];
U1q(3.422950108557157*pi,0.2228088358595417*pi) q[5];
U1q(3.51989144394989*pi,0.13685029952217698*pi) q[6];
U1q(3.789285511727258*pi,0.6219413455177519*pi) q[7];
U1q(1.80000994748434*pi,1.9966315686258298*pi) q[8];
U1q(1.64108348074806*pi,0.4818532346341209*pi) q[9];
U1q(3.237902082149618*pi,1.8941755085615646*pi) q[10];
U1q(3.865448849070157*pi,1.2382617478025817*pi) q[11];
U1q(3.235390714666838*pi,0.2822495247742107*pi) q[12];
U1q(3.741522747036553*pi,1.1031531097921636*pi) q[13];
U1q(0.673173748834378*pi,1.527461116220554*pi) q[14];
U1q(1.78519238011446*pi,1.9718713670518522*pi) q[15];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[6],q[2];
RZZ(0.5*pi) q[9],q[3];
RZZ(0.5*pi) q[4],q[11];
RZZ(0.5*pi) q[5],q[13];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[15],q[10];
U1q(1.19671920611729*pi,1.16317515905346*pi) q[0];
U1q(1.77687112886136*pi,0.56620884450109*pi) q[1];
U1q(0.373281302059252*pi,0.1518764627327167*pi) q[2];
U1q(3.405965117405302*pi,0.9212928917358326*pi) q[3];
U1q(1.68949849556189*pi,0.4482692413496565*pi) q[4];
U1q(1.47744536015367*pi,0.38852830686313666*pi) q[5];
U1q(3.158889412667561*pi,0.4161299754587868*pi) q[6];
U1q(1.7512280528028*pi,1.5292771066682296*pi) q[7];
U1q(1.11517237502124*pi,0.608480490490031*pi) q[8];
U1q(0.308772669515241*pi,0.3650836632703347*pi) q[9];
U1q(3.813720365592947*pi,1.0835394093126087*pi) q[10];
U1q(1.26781274730579*pi,0.9846967247640348*pi) q[11];
U1q(1.08783035886698*pi,1.5453168873778333*pi) q[12];
U1q(0.807844281405938*pi,0.6275595175812638*pi) q[13];
U1q(3.119608142708833*pi,0.8559924039228202*pi) q[14];
U1q(1.29216841849922*pi,0.32542178353548223*pi) q[15];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[7],q[13];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[14],q[12];
U1q(1.56272759840656*pi,1.7416717268214983*pi) q[0];
U1q(3.591022161086566*pi,0.7992800991159839*pi) q[1];
U1q(1.30570263330223*pi,0.0684252540991066*pi) q[2];
U1q(3.418555441051815*pi,1.3393486459515396*pi) q[3];
U1q(3.435578796836121*pi,1.8456291983302968*pi) q[4];
U1q(0.403765087133137*pi,1.4256460379211175*pi) q[5];
U1q(3.585244262592572*pi,0.8755916370834769*pi) q[6];
U1q(0.327305168839848*pi,0.21423892855948998*pi) q[7];
U1q(1.61879162734474*pi,1.4612944189439014*pi) q[8];
U1q(1.69392469084114*pi,0.5279776618415051*pi) q[9];
U1q(3.6570244900575632*pi,1.3897331258703916*pi) q[10];
U1q(1.23866740987271*pi,0.49595580351770474*pi) q[11];
U1q(0.365700488534244*pi,1.192831218790478*pi) q[12];
U1q(1.361302770462*pi,0.606081762434564*pi) q[13];
U1q(3.29789161711836*pi,0.9063693348324664*pi) q[14];
U1q(1.68567236925422*pi,1.1141406332638475*pi) q[15];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[15],q[14];
U1q(0.444856588752441*pi,1.2651909964792987*pi) q[0];
U1q(3.038510583757197*pi,0.3106915815510538*pi) q[1];
U1q(1.65857317207042*pi,1.181822275308364*pi) q[2];
U1q(0.540710511748001*pi,0.4081311909202445*pi) q[3];
U1q(1.50129567717926*pi,0.37706942372898933*pi) q[4];
U1q(1.54206965971047*pi,1.2428270979266265*pi) q[5];
U1q(1.68142430149795*pi,1.984164785611548*pi) q[6];
U1q(3.9357152561890447*pi,0.8371367749380497*pi) q[7];
U1q(3.65216083083457*pi,0.4537527923336695*pi) q[8];
U1q(1.64087480608187*pi,1.429313666851689*pi) q[9];
U1q(3.55247351274617*pi,0.6635418427753921*pi) q[10];
U1q(3.672121822164152*pi,1.2728397121684205*pi) q[11];
U1q(0.127831042876454*pi,0.548189448102052*pi) q[12];
U1q(1.66273212602378*pi,1.3820254025429097*pi) q[13];
U1q(3.350210759985251*pi,1.4793711749626763*pi) q[14];
U1q(1.26648104810919*pi,0.5874406187007777*pi) q[15];
RZZ(0.5*pi) q[5],q[0];
RZZ(0.5*pi) q[13],q[1];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[4],q[14];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[11],q[12];
U1q(0.649890457893242*pi,1.378892755090808*pi) q[0];
U1q(1.61220883104478*pi,0.7834456924500586*pi) q[1];
U1q(0.665425715418614*pi,0.04475700005467509*pi) q[2];
U1q(0.508542626416511*pi,0.6253344211792444*pi) q[3];
U1q(0.333888152913213*pi,1.4626636439426797*pi) q[4];
U1q(3.579559922446621*pi,1.4398464879172899*pi) q[5];
U1q(0.582079572299668*pi,1.1364922514396785*pi) q[6];
U1q(1.61515549601688*pi,0.16384531819858061*pi) q[7];
U1q(1.45666684998964*pi,0.3370871214046627*pi) q[8];
U1q(0.557901336705948*pi,1.271063295483609*pi) q[9];
U1q(1.94294151372963*pi,1.7575207762968024*pi) q[10];
U1q(1.71732939252299*pi,0.9297056674468154*pi) q[11];
U1q(0.305689800173609*pi,1.1724917632271419*pi) q[12];
U1q(0.382874933404271*pi,1.7106072692991692*pi) q[13];
U1q(3.494195371451958*pi,1.7715895501041548*pi) q[14];
U1q(1.58277063696739*pi,0.22087811811763025*pi) q[15];
rz(2.621107244909192*pi) q[0];
rz(1.2165543075499414*pi) q[1];
rz(1.955242999945325*pi) q[2];
rz(3.3746655788207556*pi) q[3];
rz(0.5373363560573203*pi) q[4];
rz(2.56015351208271*pi) q[5];
rz(0.8635077485603215*pi) q[6];
rz(3.8361546818014194*pi) q[7];
rz(1.6629128785953373*pi) q[8];
rz(0.728936704516391*pi) q[9];
rz(2.2424792237031976*pi) q[10];
rz(3.0702943325531846*pi) q[11];
rz(0.8275082367728581*pi) q[12];
rz(0.2893927307008308*pi) q[13];
rz(2.2284104498958452*pi) q[14];
rz(1.7791218818823697*pi) q[15];
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