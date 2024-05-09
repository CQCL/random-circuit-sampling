OPENQASM 2.0;
include "hqslib1.inc";

qreg q[24];
creg c[24];
U1q(0.594994341496379*pi,0.885824108008021*pi) q[0];
U1q(0.304654653885102*pi,0.88683669423203*pi) q[1];
U1q(0.754820369098283*pi,0.141677185671436*pi) q[2];
U1q(0.700498516569886*pi,1.533850161312684*pi) q[3];
U1q(0.405794371075626*pi,0.101394775268219*pi) q[4];
U1q(0.570418460639282*pi,1.007937529248742*pi) q[5];
U1q(0.669260642871746*pi,0.452719024005059*pi) q[6];
U1q(0.388075835465277*pi,0.655730955901847*pi) q[7];
U1q(0.514435406215459*pi,0.328053912934718*pi) q[8];
U1q(0.41745146343553*pi,0.833037205177909*pi) q[9];
U1q(0.419511504103846*pi,0.133801143122737*pi) q[10];
U1q(0.229581950210049*pi,0.0133450133718267*pi) q[11];
U1q(0.829591662426111*pi,0.274037530830005*pi) q[12];
U1q(0.545145191020476*pi,0.293598615244956*pi) q[13];
U1q(0.24679805219735*pi,1.747940253496061*pi) q[14];
U1q(0.669270077682178*pi,1.40619303909425*pi) q[15];
U1q(0.126792294336632*pi,0.542576624947876*pi) q[16];
U1q(0.418653480688278*pi,0.310690046333*pi) q[17];
U1q(0.454650630020236*pi,0.53163856324749*pi) q[18];
U1q(0.0551267993802038*pi,1.01593416668249*pi) q[19];
U1q(0.293047558589188*pi,1.714480876186014*pi) q[20];
U1q(0.312302617426004*pi,0.20695115811075*pi) q[21];
U1q(0.893318813753989*pi,0.434414621507888*pi) q[22];
U1q(0.286265500997467*pi,0.61169425760271*pi) q[23];
RZZ(0.5*pi) q[10],q[0];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[17],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[6],q[12];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[21],q[18];
U1q(0.182927318842723*pi,0.9018555759797999*pi) q[0];
U1q(0.295254495959646*pi,1.2332530061988298*pi) q[1];
U1q(0.429441405397081*pi,0.93681934216767*pi) q[2];
U1q(0.267910711190023*pi,0.002549048873010218*pi) q[3];
U1q(0.210137480364722*pi,0.20591337023445*pi) q[4];
U1q(0.866477088829193*pi,0.9124044237181699*pi) q[5];
U1q(0.724868013004018*pi,1.91031647019005*pi) q[6];
U1q(0.28127326723078*pi,1.61410935220669*pi) q[7];
U1q(0.669074718902072*pi,0.1701838634352899*pi) q[8];
U1q(0.725757319753356*pi,1.3874683070157618*pi) q[9];
U1q(0.485033759298249*pi,1.95993217456662*pi) q[10];
U1q(0.307992071330561*pi,1.2893271688192298*pi) q[11];
U1q(0.455416416070162*pi,1.9978578633533899*pi) q[12];
U1q(0.332247543799908*pi,0.014862882062190108*pi) q[13];
U1q(0.492841417651336*pi,0.042713791732440054*pi) q[14];
U1q(0.778026653757269*pi,1.9072511131195178*pi) q[15];
U1q(0.202238330160319*pi,0.64982834975325*pi) q[16];
U1q(0.361389773066379*pi,1.2336309165171402*pi) q[17];
U1q(0.637714745691158*pi,0.36158652552229986*pi) q[18];
U1q(0.361453508974934*pi,1.22332547799846*pi) q[19];
U1q(0.363985468323512*pi,0.2595383842944401*pi) q[20];
U1q(0.748217772376478*pi,1.8103490791633199*pi) q[21];
U1q(0.45955573554036*pi,1.96677074946974*pi) q[22];
U1q(0.810335762133153*pi,0.9029692481297298*pi) q[23];
RZZ(0.5*pi) q[6],q[0];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[18],q[4];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[17],q[9];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[21],q[14];
RZZ(0.5*pi) q[19],q[20];
RZZ(0.5*pi) q[22],q[23];
U1q(0.339225074933654*pi,0.14824514407062006*pi) q[0];
U1q(0.201044950982011*pi,1.9901024914262102*pi) q[1];
U1q(0.729884134423805*pi,0.5074866495298398*pi) q[2];
U1q(0.753329392906017*pi,0.04666856040496015*pi) q[3];
U1q(0.526136662280259*pi,0.8587986598555704*pi) q[4];
U1q(0.493352978922026*pi,1.89902653166026*pi) q[5];
U1q(0.604508252374063*pi,1.3194592751767598*pi) q[6];
U1q(0.702821971154406*pi,1.1806829750490797*pi) q[7];
U1q(0.843046547548767*pi,1.5389385161132596*pi) q[8];
U1q(0.635646730980898*pi,1.7194852816631099*pi) q[9];
U1q(0.590867986453943*pi,0.7344103626488403*pi) q[10];
U1q(0.259510126538162*pi,0.33230966296583997*pi) q[11];
U1q(0.247113734449878*pi,0.1556447051073402*pi) q[12];
U1q(0.754202694016742*pi,0.36334169782861014*pi) q[13];
U1q(0.443919147873664*pi,0.8657722518977398*pi) q[14];
U1q(0.204295505891406*pi,1.25340386476265*pi) q[15];
U1q(0.60765391843811*pi,0.3682293322253698*pi) q[16];
U1q(0.631276152516855*pi,0.4737382162535102*pi) q[17];
U1q(0.266386633331105*pi,0.5921288823150803*pi) q[18];
U1q(0.645957943112429*pi,1.2868684218351998*pi) q[19];
U1q(0.444864653013742*pi,0.44977381750387035*pi) q[20];
U1q(0.600861794206576*pi,1.8065920939519202*pi) q[21];
U1q(0.780460237396741*pi,0.14472520854933002*pi) q[22];
U1q(0.621918139882857*pi,1.09388154350043*pi) q[23];
RZZ(0.5*pi) q[0],q[4];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[19],q[2];
RZZ(0.5*pi) q[6],q[3];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[9],q[7];
RZZ(0.5*pi) q[18],q[8];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[14];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[23],q[20];
U1q(0.789103516899266*pi,1.4324136450888103*pi) q[0];
U1q(0.637030667784243*pi,0.8347898808366203*pi) q[1];
U1q(0.598011319601418*pi,0.2727886176101997*pi) q[2];
U1q(0.838048610363221*pi,0.056989782968940084*pi) q[3];
U1q(0.511424013004138*pi,1.6013610988096296*pi) q[4];
U1q(0.562299171467825*pi,0.7090561093750702*pi) q[5];
U1q(0.684009666448949*pi,1.3669182987285797*pi) q[6];
U1q(0.144349276084642*pi,1.70400700508132*pi) q[7];
U1q(0.249836589262435*pi,0.5120516028106596*pi) q[8];
U1q(0.562749929434691*pi,1.9602455959542402*pi) q[9];
U1q(0.407294894839381*pi,0.81033543666073*pi) q[10];
U1q(0.591877571887397*pi,1.3011844062303997*pi) q[11];
U1q(0.579607517738881*pi,0.45074547191897985*pi) q[12];
U1q(0.365366548581485*pi,0.17083978509590025*pi) q[13];
U1q(0.672701714501309*pi,0.8345651026405498*pi) q[14];
U1q(0.777105479355184*pi,0.7065063654596102*pi) q[15];
U1q(0.609107031511665*pi,1.7695711166674801*pi) q[16];
U1q(0.28128395885242*pi,0.5111511358958403*pi) q[17];
U1q(0.633457902916461*pi,0.09871111790070053*pi) q[18];
U1q(0.480713275002245*pi,0.7560224536978399*pi) q[19];
U1q(0.430220406639285*pi,1.0958100932732302*pi) q[20];
U1q(0.559693199297511*pi,0.7089515705324398*pi) q[21];
U1q(0.78688352803454*pi,0.15916000242623962*pi) q[22];
U1q(0.378075657243931*pi,1.57194404782497*pi) q[23];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[23],q[4];
RZZ(0.5*pi) q[22],q[7];
RZZ(0.5*pi) q[8],q[12];
RZZ(0.5*pi) q[18],q[9];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[15],q[13];
RZZ(0.5*pi) q[19],q[14];
RZZ(0.5*pi) q[21],q[16];
U1q(0.538774530711444*pi,1.0183441734647003*pi) q[0];
U1q(0.0956942848477346*pi,1.0682857771942*pi) q[1];
U1q(0.174013387828536*pi,0.047119689745679594*pi) q[2];
U1q(0.424754265242609*pi,0.5256286335981493*pi) q[3];
U1q(0.0198433046451732*pi,0.7166415908741008*pi) q[4];
U1q(0.249950476864177*pi,1.1174060000200008*pi) q[5];
U1q(0.62080159804042*pi,0.6819474105091397*pi) q[6];
U1q(0.881910553292859*pi,1.3808543778263296*pi) q[7];
U1q(0.192250913786775*pi,1.7068931574532993*pi) q[8];
U1q(0.477123635945458*pi,1.9694057257946502*pi) q[9];
U1q(0.583073890949073*pi,0.5450381517712897*pi) q[10];
U1q(0.19576230193972*pi,1.00732444326208*pi) q[11];
U1q(0.547958188830643*pi,0.39862353777061976*pi) q[12];
U1q(0.494510509230121*pi,0.5377335324069001*pi) q[13];
U1q(0.2315602973523*pi,1.5937761580271008*pi) q[14];
U1q(0.473118414458848*pi,1.9169407189951597*pi) q[15];
U1q(0.564711399671257*pi,1.8655426658940009*pi) q[16];
U1q(0.211870170596344*pi,0.7106988561116108*pi) q[17];
U1q(0.950293657264871*pi,1.5572222777056002*pi) q[18];
U1q(0.762311103470813*pi,0.9697563893443899*pi) q[19];
U1q(0.197992382576467*pi,1.9131360107425994*pi) q[20];
U1q(0.916154405222445*pi,0.9716835447438203*pi) q[21];
U1q(0.394240928751462*pi,0.3641860419032197*pi) q[22];
U1q(0.536045695834889*pi,1.6048514963079406*pi) q[23];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[17],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[7];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[14];
RZZ(0.5*pi) q[8],q[20];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[19],q[12];
RZZ(0.5*pi) q[23],q[16];
RZZ(0.5*pi) q[18],q[22];
U1q(0.77864390233365*pi,1.0701042074367102*pi) q[0];
U1q(0.758802330707983*pi,1.5736477127887003*pi) q[1];
U1q(0.455233587284795*pi,0.3470764051389992*pi) q[2];
U1q(0.958760600891748*pi,0.9815646488087992*pi) q[3];
U1q(0.230046700690858*pi,0.9150608530454001*pi) q[4];
U1q(0.552968281916148*pi,1.7933900183018991*pi) q[5];
U1q(0.761365703074694*pi,0.06848276262163999*pi) q[6];
U1q(0.70912730823969*pi,0.1453285857398008*pi) q[7];
U1q(0.675728254682214*pi,1.7472867629755005*pi) q[8];
U1q(0.839082303582599*pi,1.5674140633138691*pi) q[9];
U1q(0.418536692069927*pi,0.8218266892492494*pi) q[10];
U1q(0.0492175607800136*pi,1.4949747166565004*pi) q[11];
U1q(0.742194808106306*pi,0.65645245072559*pi) q[12];
U1q(0.687776436941975*pi,1.6894314763687994*pi) q[13];
U1q(0.377633941927677*pi,0.5881461505421992*pi) q[14];
U1q(0.817159956798242*pi,0.9465175114745801*pi) q[15];
U1q(0.647203627665471*pi,1.2379996394983*pi) q[16];
U1q(0.497246948569659*pi,0.5577807993208008*pi) q[17];
U1q(0.342594599848062*pi,1.9528353727232002*pi) q[18];
U1q(0.8188928016659*pi,1.6651500861332007*pi) q[19];
U1q(0.824228922878988*pi,1.5673234564899001*pi) q[20];
U1q(0.635487758752422*pi,1.4472806258874993*pi) q[21];
U1q(0.447169523579555*pi,1.8006293033025997*pi) q[22];
U1q(0.251178120300264*pi,1.1677831650558694*pi) q[23];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[11],q[12];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[19],q[16];
U1q(0.76811683161173*pi,1.3262573651213998*pi) q[0];
U1q(0.130320827273861*pi,0.7501474752573003*pi) q[1];
U1q(0.735702311860494*pi,0.02057810811220051*pi) q[2];
U1q(0.228846686920108*pi,1.9295704644429996*pi) q[3];
U1q(0.525424036488644*pi,0.27731373722179953*pi) q[4];
U1q(0.503798588102586*pi,0.15818120872189922*pi) q[5];
U1q(0.89729117977582*pi,1.5915079960868006*pi) q[6];
U1q(0.731015224663976*pi,1.1879527872702003*pi) q[7];
U1q(0.661987639679912*pi,1.4143737375672991*pi) q[8];
U1q(0.44758543689388*pi,1.9435206442064992*pi) q[9];
U1q(0.318135447874359*pi,0.8110762006654006*pi) q[10];
U1q(0.657993277452225*pi,1.2108331305809*pi) q[11];
U1q(0.790063747253385*pi,0.5748050448793993*pi) q[12];
U1q(0.815422251735302*pi,0.15963722983189932*pi) q[13];
U1q(0.560896272589548*pi,1.5990159196960008*pi) q[14];
U1q(0.377652251395062*pi,0.8750996376129496*pi) q[15];
U1q(0.412976999335472*pi,0.16484811211349992*pi) q[16];
U1q(0.708837950417699*pi,0.5549581213811994*pi) q[17];
U1q(0.273362495305459*pi,1.9720353650356017*pi) q[18];
U1q(0.730194751797416*pi,1.9260117100527996*pi) q[19];
U1q(0.84433781018147*pi,0.10850444164769968*pi) q[20];
U1q(0.0744248293754144*pi,0.9464478345966008*pi) q[21];
U1q(0.311696433456267*pi,1.3168215232922993*pi) q[22];
U1q(0.165954636586462*pi,0.7176933372034*pi) q[23];
RZZ(0.5*pi) q[0],q[20];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[16];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[21],q[17];
RZZ(0.5*pi) q[18],q[23];
U1q(0.141470381816619*pi,0.5367146093932007*pi) q[0];
U1q(0.303985799212108*pi,0.21684141684389857*pi) q[1];
U1q(0.710698959944958*pi,1.7898870434499017*pi) q[2];
U1q(0.6126661230917*pi,1.6970190017868987*pi) q[3];
U1q(0.681479009115356*pi,0.3146148621509006*pi) q[4];
U1q(0.463728711716943*pi,0.9430079547625994*pi) q[5];
U1q(0.57322854370028*pi,1.9086486274586996*pi) q[6];
U1q(0.852534202967046*pi,0.37957811194760005*pi) q[7];
U1q(0.95614052338031*pi,0.20105818997670077*pi) q[8];
U1q(0.367943728751944*pi,1.3810284150361998*pi) q[9];
U1q(0.0527881794884808*pi,1.3149971909308*pi) q[10];
U1q(0.680778666699315*pi,0.5543605346727993*pi) q[11];
U1q(0.514686920766863*pi,0.7375728683835003*pi) q[12];
U1q(0.654242068558289*pi,0.2736309184010004*pi) q[13];
U1q(0.147180219046443*pi,1.222858694503799*pi) q[14];
U1q(0.42348205424698*pi,1.1768903694343997*pi) q[15];
U1q(0.38079422583807*pi,1.3402172613672008*pi) q[16];
U1q(0.837090366244249*pi,0.3621674922429001*pi) q[17];
U1q(0.487676170880562*pi,1.5677203988447985*pi) q[18];
U1q(0.633231604269017*pi,0.3905252235283001*pi) q[19];
U1q(0.525897982605384*pi,1.3488750925302995*pi) q[20];
U1q(0.434004703449139*pi,0.06209493675450162*pi) q[21];
U1q(0.360962485735828*pi,1.5658891119696996*pi) q[22];
U1q(0.483809301691427*pi,0.5144350477781998*pi) q[23];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[16],q[1];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[5],q[4];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[19],q[7];
RZZ(0.5*pi) q[22],q[9];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[13],q[12];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[18],q[17];
U1q(0.2150589740238*pi,1.252267651421299*pi) q[0];
U1q(0.481321584453784*pi,0.17683663561529883*pi) q[1];
U1q(0.634175311347999*pi,0.5758104569176012*pi) q[2];
U1q(0.325413158470666*pi,0.22506673850460146*pi) q[3];
U1q(0.493511645404914*pi,1.6855881950451987*pi) q[4];
U1q(0.306222441309924*pi,1.4449927851876012*pi) q[5];
U1q(0.488526959700294*pi,0.6618741475258005*pi) q[6];
U1q(0.337102384795446*pi,0.8491541827352016*pi) q[7];
U1q(0.653886560705844*pi,1.0553814528339984*pi) q[8];
U1q(0.559690650196685*pi,0.43626453415400057*pi) q[9];
U1q(0.388883623263499*pi,0.8298121421825009*pi) q[10];
U1q(0.990076980249363*pi,1.668147528659901*pi) q[11];
U1q(0.330393512016247*pi,1.654939607706*pi) q[12];
U1q(0.111584166334182*pi,1.9744813166221995*pi) q[13];
U1q(0.595619925482878*pi,0.2686367524730997*pi) q[14];
U1q(0.76899919087457*pi,1.3755313255954995*pi) q[15];
U1q(0.46654616828342*pi,0.8623020881520986*pi) q[16];
U1q(0.519659539797093*pi,1.6320599245274998*pi) q[17];
U1q(0.952073599163174*pi,1.5002760874189*pi) q[18];
U1q(0.453117081475467*pi,1.5085177037380006*pi) q[19];
U1q(0.471326189343321*pi,1.472187888283301*pi) q[20];
U1q(0.229823127501742*pi,1.668962909481099*pi) q[21];
U1q(0.0809430974168455*pi,0.17953463212590037*pi) q[22];
U1q(0.108605507132668*pi,1.9924438754529987*pi) q[23];
RZZ(0.5*pi) q[18],q[0];
RZZ(0.5*pi) q[8],q[1];
RZZ(0.5*pi) q[2],q[23];
RZZ(0.5*pi) q[21],q[3];
RZZ(0.5*pi) q[4],q[14];
RZZ(0.5*pi) q[5],q[16];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[17],q[20];
RZZ(0.5*pi) q[22],q[19];
U1q(0.504006691775709*pi,0.20662833701830152*pi) q[0];
U1q(0.709225317992778*pi,0.27069716599530125*pi) q[1];
U1q(0.193350403044293*pi,1.8578574813653006*pi) q[2];
U1q(0.583721584792583*pi,0.6867067685180004*pi) q[3];
U1q(0.841901853119996*pi,0.3439567663948999*pi) q[4];
U1q(0.371657647063631*pi,0.7719125143940992*pi) q[5];
U1q(0.653252682675277*pi,0.2154867959655995*pi) q[6];
U1q(0.813527412278532*pi,0.6729485751201985*pi) q[7];
U1q(0.538841744601213*pi,0.8239650826096003*pi) q[8];
U1q(0.450373433678663*pi,1.4862106746893993*pi) q[9];
U1q(0.407385077762111*pi,1.0122398518014002*pi) q[10];
U1q(0.650237875469373*pi,0.7369731526535013*pi) q[11];
U1q(0.489133835942593*pi,1.5143193056247988*pi) q[12];
U1q(0.66092418165061*pi,1.9588726159160998*pi) q[13];
U1q(0.527163549657191*pi,1.3532441059648015*pi) q[14];
U1q(0.578248374660271*pi,1.9400513617584991*pi) q[15];
U1q(0.533138750070218*pi,1.1505905273839012*pi) q[16];
U1q(0.665551776469856*pi,1.422355125816999*pi) q[17];
U1q(0.9356859456578*pi,0.09223701671210094*pi) q[18];
U1q(0.135812934676204*pi,1.587834359499201*pi) q[19];
U1q(0.376330053776379*pi,0.8926413719653006*pi) q[20];
U1q(0.675722360397986*pi,1.2234258708497983*pi) q[21];
U1q(0.913127226210232*pi,0.16455689513470162*pi) q[22];
U1q(0.378455326926751*pi,1.5294228439939985*pi) q[23];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[6],q[2];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[5],q[14];
RZZ(0.5*pi) q[7],q[16];
RZZ(0.5*pi) q[13],q[8];
RZZ(0.5*pi) q[9],q[20];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[21],q[22];
U1q(0.372257026283818*pi,0.8266198140905985*pi) q[0];
U1q(0.446288165676026*pi,0.34496950573869967*pi) q[1];
U1q(0.459614388409938*pi,0.6483750851130985*pi) q[2];
U1q(0.629978066147357*pi,1.8255152544262998*pi) q[3];
U1q(0.641378310782526*pi,1.1720408460729992*pi) q[4];
U1q(0.37435950477117*pi,0.24708244014719938*pi) q[5];
U1q(0.897447465524052*pi,1.4851039201919995*pi) q[6];
U1q(0.470307852692433*pi,1.8321174669595983*pi) q[7];
U1q(0.398939839535402*pi,1.5141310826710992*pi) q[8];
U1q(0.75469011772406*pi,0.6971604593354002*pi) q[9];
U1q(0.238863504266413*pi,1.6891629719718004*pi) q[10];
U1q(0.356589010331389*pi,0.2736294395268004*pi) q[11];
U1q(0.58570497050845*pi,1.5308699274516009*pi) q[12];
U1q(0.175535344164418*pi,1.9967890686512*pi) q[13];
U1q(0.72106080904008*pi,1.170399286126699*pi) q[14];
U1q(0.312343543546984*pi,0.40764715822379927*pi) q[15];
U1q(0.492940808439482*pi,1.4229936204358005*pi) q[16];
U1q(0.137317606828352*pi,1.5091605735001004*pi) q[17];
U1q(0.631699704403243*pi,1.0272710675470016*pi) q[18];
U1q(0.269362463886244*pi,1.5044691384551*pi) q[19];
U1q(0.349191807790562*pi,0.7092129897036017*pi) q[20];
U1q(0.358014520369804*pi,0.14398531147369908*pi) q[21];
U1q(0.466997599812962*pi,0.39713835558499966*pi) q[22];
U1q(0.597065972279969*pi,1.3462256347334005*pi) q[23];
RZZ(0.5*pi) q[19],q[0];
RZZ(0.5*pi) q[9],q[1];
RZZ(0.5*pi) q[2],q[7];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[15],q[4];
RZZ(0.5*pi) q[6],q[5];
RZZ(0.5*pi) q[10],q[8];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[17],q[12];
RZZ(0.5*pi) q[16],q[14];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[21],q[23];
U1q(0.205345903025131*pi,0.8628523766534997*pi) q[0];
U1q(0.495025901186078*pi,0.024950077606199272*pi) q[1];
U1q(0.555525086461677*pi,1.5557924698300987*pi) q[2];
U1q(0.731032939057404*pi,1.4979139836427002*pi) q[3];
U1q(0.900175672158373*pi,1.9903201669065993*pi) q[4];
U1q(0.591657279139858*pi,1.0635703478824006*pi) q[5];
U1q(0.577791581972354*pi,1.0733409858111997*pi) q[6];
U1q(0.146600862660603*pi,0.3047396229369994*pi) q[7];
U1q(0.514287954202529*pi,1.1196207131733011*pi) q[8];
U1q(0.653828384676048*pi,1.7870607062516015*pi) q[9];
U1q(0.407509211779217*pi,0.30871508031630057*pi) q[10];
U1q(0.438665318181648*pi,1.9789064539155987*pi) q[11];
U1q(0.604009437937887*pi,1.468121373634201*pi) q[12];
U1q(0.229519031898091*pi,0.8040170770599993*pi) q[13];
U1q(0.56250485147783*pi,1.7233260286036014*pi) q[14];
U1q(0.5577856621972*pi,1.126836039105001*pi) q[15];
U1q(0.406743063422118*pi,1.5513858213284983*pi) q[16];
U1q(0.263020142433904*pi,1.4138429963138002*pi) q[17];
U1q(0.351111957912992*pi,1.7748316004833988*pi) q[18];
U1q(0.712863889094943*pi,0.6237240006703004*pi) q[19];
U1q(0.738005074956102*pi,0.6226990341145004*pi) q[20];
U1q(0.422651445299782*pi,1.7467536031192985*pi) q[21];
U1q(0.770794533063752*pi,1.6958697123396007*pi) q[22];
U1q(0.785344461474037*pi,0.26029769562619975*pi) q[23];
RZZ(0.5*pi) q[0],q[16];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[22],q[2];
RZZ(0.5*pi) q[18],q[3];
RZZ(0.5*pi) q[6],q[4];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[21],q[7];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[10],q[12];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[15],q[20];
U1q(0.386737399055101*pi,1.410481457259099*pi) q[0];
U1q(0.425109257667512*pi,1.6250971052308003*pi) q[1];
U1q(0.577739452739163*pi,1.3383251620834997*pi) q[2];
U1q(0.871547730291696*pi,1.3704521804957004*pi) q[3];
U1q(0.361438096665188*pi,0.8798026890552997*pi) q[4];
U1q(0.361839583471983*pi,0.9195879107696001*pi) q[5];
U1q(0.426669852197697*pi,1.6552553454405015*pi) q[6];
U1q(0.597486491867348*pi,0.04393983880190078*pi) q[7];
U1q(0.263723026406898*pi,0.7639405032347995*pi) q[8];
U1q(0.176156651174286*pi,1.038743417285101*pi) q[9];
U1q(0.442916661619874*pi,1.9365211330973011*pi) q[10];
U1q(0.896249791171826*pi,0.6097447773410991*pi) q[11];
U1q(0.428057794200377*pi,0.6657506149614001*pi) q[12];
U1q(0.38377739816673*pi,1.8821506458607011*pi) q[13];
U1q(0.470961234448691*pi,1.2059992754540012*pi) q[14];
U1q(0.335117146222923*pi,0.20149737510120147*pi) q[15];
U1q(0.207231658427365*pi,0.3066201192925*pi) q[16];
U1q(0.184797406807848*pi,1.7147317087379008*pi) q[17];
U1q(0.342675479894045*pi,1.0276664207437989*pi) q[18];
U1q(0.113548901544617*pi,0.8462009078020003*pi) q[19];
U1q(0.358534045975563*pi,0.7101110038764986*pi) q[20];
U1q(0.711991113993491*pi,0.6842888821429014*pi) q[21];
U1q(0.203558843145731*pi,0.4143280751566998*pi) q[22];
U1q(0.631056541547633*pi,1.1927614710383985*pi) q[23];
rz(0.21820168087729996*pi) q[0];
rz(1.7168919068583008*pi) q[1];
rz(0.35656183085669824*pi) q[2];
rz(3.6228880072018015*pi) q[3];
rz(2.639414260578299*pi) q[4];
rz(1.2454636210451007*pi) q[5];
rz(2.4922383670239014*pi) q[6];
rz(3.9917360295761988*pi) q[7];
rz(3.115282108941699*pi) q[8];
rz(2.238820016756499*pi) q[9];
rz(2.2235314777251993*pi) q[10];
rz(2.9314579437366*pi) q[11];
rz(3.5297685154396987*pi) q[12];
rz(2.3932166231534993*pi) q[13];
rz(0.03806796879180041*pi) q[14];
rz(2.9791809335011017*pi) q[15];
rz(3.0779644575933*pi) q[16];
rz(3.5193053820966007*pi) q[17];
rz(0.9205435301627034*pi) q[18];
rz(2.254400460809201*pi) q[19];
rz(2.2713338465870017*pi) q[20];
rz(2.7737859449919*pi) q[21];
rz(2.5224402327792*pi) q[22];
rz(0.8956214231410016*pi) q[23];
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