OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.732966246251694*pi,1.621241289530332*pi) q[0];
U1q(0.361191546549451*pi,0.853518087808675*pi) q[1];
U1q(0.362465785072262*pi,0.847544000087911*pi) q[2];
U1q(0.58023197381199*pi,1.231826138919782*pi) q[3];
U1q(0.213227357831299*pi,0.6801894244533699*pi) q[4];
U1q(0.494045443763158*pi,0.655427187142151*pi) q[5];
U1q(0.348951583451428*pi,1.590609314865924*pi) q[6];
U1q(0.500387084425587*pi,1.739414859608176*pi) q[7];
U1q(0.19499201874084*pi,0.38487368979363*pi) q[8];
U1q(0.632088986578739*pi,1.703504022510372*pi) q[9];
U1q(0.586799473897257*pi,1.713807518183148*pi) q[10];
U1q(0.501902594585289*pi,0.785980840993941*pi) q[11];
U1q(0.983130515807091*pi,0.529375824075975*pi) q[12];
U1q(0.224386804634395*pi,1.888378510280329*pi) q[13];
U1q(0.507808435876832*pi,0.0386530941848992*pi) q[14];
U1q(0.557344231299322*pi,1.1757898397871*pi) q[15];
U1q(0.618866300634173*pi,1.39802630448605*pi) q[16];
U1q(0.374598310839938*pi,0.8650753182219499*pi) q[17];
U1q(0.299776128239473*pi,0.0331122675407091*pi) q[18];
U1q(0.276530162317918*pi,1.377941760713451*pi) q[19];
U1q(0.382834467857115*pi,0.293853224607565*pi) q[20];
U1q(0.640644926456729*pi,1.23529069600662*pi) q[21];
U1q(0.572496910543961*pi,0.214760120033942*pi) q[22];
U1q(0.824495510512456*pi,0.609132218489925*pi) q[23];
U1q(0.151327533481392*pi,0.650381686258114*pi) q[24];
U1q(0.684593544710341*pi,0.21628166999444*pi) q[25];
U1q(0.834380129964702*pi,1.5483419664005709*pi) q[26];
U1q(0.619816579611219*pi,1.115166748634649*pi) q[27];
U1q(0.157242263544782*pi,1.885655399286038*pi) q[28];
U1q(0.531259743032892*pi,0.714161147106654*pi) q[29];
U1q(0.694081914489442*pi,0.615022505108601*pi) q[30];
U1q(0.821236818320878*pi,0.687335475730095*pi) q[31];
U1q(0.416634425444636*pi,1.108423868801521*pi) q[32];
U1q(0.428970943088026*pi,0.9180585733062401*pi) q[33];
U1q(0.618949785639306*pi,1.568541301073372*pi) q[34];
U1q(0.695875730293668*pi,1.11864382927748*pi) q[35];
U1q(0.132674203689853*pi,0.276429637384668*pi) q[36];
U1q(0.660667837866444*pi,1.139954710947825*pi) q[37];
U1q(0.603830324486722*pi,0.9932997712287199*pi) q[38];
U1q(0.297330770623419*pi,0.83115643337543*pi) q[39];
RZZ(0.5*pi) q[0],q[35];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[2],q[26];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[13],q[6];
RZZ(0.5*pi) q[25],q[7];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[32],q[9];
RZZ(0.5*pi) q[10],q[38];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[33],q[19];
RZZ(0.5*pi) q[37],q[22];
RZZ(0.5*pi) q[28],q[27];
RZZ(0.5*pi) q[39],q[29];
RZZ(0.5*pi) q[36],q[31];
U1q(0.670685078978409*pi,0.47104830907805995*pi) q[0];
U1q(0.241867859465796*pi,0.59044826555675*pi) q[1];
U1q(0.55469416396014*pi,0.7989290167393299*pi) q[2];
U1q(0.461979186616223*pi,0.2045856421290102*pi) q[3];
U1q(0.248435033943604*pi,1.47002440133594*pi) q[4];
U1q(0.835428552379391*pi,1.0515567731188011*pi) q[5];
U1q(0.487598937999006*pi,1.11153523061581*pi) q[6];
U1q(0.483468764387583*pi,1.9322925157610298*pi) q[7];
U1q(0.51159259650066*pi,0.3649907637046601*pi) q[8];
U1q(0.552232340050984*pi,0.59875214162295*pi) q[9];
U1q(0.447200981725038*pi,0.8777069279962202*pi) q[10];
U1q(0.392390730656041*pi,0.84207769755858*pi) q[11];
U1q(0.67990572717251*pi,0.71506418380617*pi) q[12];
U1q(0.240606982081836*pi,0.3884876922796998*pi) q[13];
U1q(0.282848290785984*pi,1.7831746285919499*pi) q[14];
U1q(0.376519902740816*pi,1.197577069746077*pi) q[15];
U1q(0.190671184771264*pi,0.63332106193588*pi) q[16];
U1q(0.701618406690217*pi,1.53523335667502*pi) q[17];
U1q(0.275952839050765*pi,1.3400283248489*pi) q[18];
U1q(0.376526861114402*pi,0.060281663192019996*pi) q[19];
U1q(0.43710545429368*pi,1.68448427416868*pi) q[20];
U1q(0.103881492171158*pi,1.89908456939277*pi) q[21];
U1q(0.925619291805268*pi,0.8206643216519001*pi) q[22];
U1q(0.5231188194947*pi,0.7954523674671901*pi) q[23];
U1q(0.436497234878857*pi,0.13509018244772997*pi) q[24];
U1q(0.326109777797862*pi,0.92466294623998*pi) q[25];
U1q(0.111670360731371*pi,0.7369418601123998*pi) q[26];
U1q(0.425002174968857*pi,1.3306641452643202*pi) q[27];
U1q(0.518236012880133*pi,1.17637706149439*pi) q[28];
U1q(0.939707960523648*pi,1.064228804778385*pi) q[29];
U1q(0.0839398119766695*pi,0.72338301761116*pi) q[30];
U1q(0.196567696806649*pi,0.21714797111567*pi) q[31];
U1q(0.721864991344453*pi,0.7594972672559899*pi) q[32];
U1q(0.322771499935494*pi,0.7698167463828902*pi) q[33];
U1q(0.278735571780222*pi,1.0746661850570502*pi) q[34];
U1q(0.252734909144576*pi,1.66569537626233*pi) q[35];
U1q(0.214260475737274*pi,1.07202217245844*pi) q[36];
U1q(0.512233253807568*pi,0.31623808013423993*pi) q[37];
U1q(0.440808264069973*pi,0.23836348864577994*pi) q[38];
U1q(0.638152918877855*pi,1.82575717758818*pi) q[39];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[38],q[1];
RZZ(0.5*pi) q[2],q[17];
RZZ(0.5*pi) q[30],q[3];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[5],q[24];
RZZ(0.5*pi) q[9],q[6];
RZZ(0.5*pi) q[20],q[7];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[10],q[37];
RZZ(0.5*pi) q[32],q[11];
RZZ(0.5*pi) q[34],q[12];
RZZ(0.5*pi) q[13],q[35];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[22],q[18];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[23],q[26];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[36],q[29];
U1q(0.48374380877708*pi,1.01052005482424*pi) q[0];
U1q(0.65255421945302*pi,0.32323215851076004*pi) q[1];
U1q(0.407093952518507*pi,0.7188974819537202*pi) q[2];
U1q(0.723202462613134*pi,0.4585175503172101*pi) q[3];
U1q(0.921938853928164*pi,0.21921327371823995*pi) q[4];
U1q(0.473127922879678*pi,0.9109234129184198*pi) q[5];
U1q(0.941781471224881*pi,0.2330524054143197*pi) q[6];
U1q(0.786471482731164*pi,0.9340220463535296*pi) q[7];
U1q(0.392842289480183*pi,1.8652054054626896*pi) q[8];
U1q(0.378885254294922*pi,1.6065634511724003*pi) q[9];
U1q(0.754688616166328*pi,0.8211096637616997*pi) q[10];
U1q(0.634125532281668*pi,1.76003276956373*pi) q[11];
U1q(0.200643204201467*pi,0.9166174113727701*pi) q[12];
U1q(0.708140730623608*pi,1.63020246173084*pi) q[13];
U1q(0.594654703969641*pi,0.28631616413700023*pi) q[14];
U1q(0.462463947253318*pi,1.4566405187757199*pi) q[15];
U1q(0.52624715698431*pi,0.6092934881143601*pi) q[16];
U1q(0.147093002415427*pi,0.15617899034997018*pi) q[17];
U1q(0.153288040399059*pi,1.0691107014635204*pi) q[18];
U1q(0.634941014007873*pi,1.5125449394919803*pi) q[19];
U1q(0.643930174953509*pi,0.33477245585494986*pi) q[20];
U1q(0.078824234636716*pi,0.17680414643819997*pi) q[21];
U1q(0.723317957179194*pi,0.38006230404505015*pi) q[22];
U1q(0.377804699100584*pi,1.0545154815366198*pi) q[23];
U1q(0.575851896397406*pi,1.43750502670812*pi) q[24];
U1q(0.21139312517405*pi,1.55854333057826*pi) q[25];
U1q(0.367472323256409*pi,1.7371633025977404*pi) q[26];
U1q(0.616368340916797*pi,0.8066682112300603*pi) q[27];
U1q(0.616956278440063*pi,1.7241149065247097*pi) q[28];
U1q(0.0714171283413595*pi,1.04103635048171*pi) q[29];
U1q(0.652136689525913*pi,1.40554020326964*pi) q[30];
U1q(0.292545093204489*pi,0.11587669297620007*pi) q[31];
U1q(0.427881558817215*pi,1.9093812038256601*pi) q[32];
U1q(0.363646624414151*pi,1.1350166781755*pi) q[33];
U1q(0.599560523075488*pi,1.8592190707163603*pi) q[34];
U1q(0.703644524367747*pi,0.07793303664298978*pi) q[35];
U1q(0.77105263069075*pi,1.116586881*pi) q[36];
U1q(0.552064918978013*pi,1.2859777493072002*pi) q[37];
U1q(0.283145754663396*pi,1.2730382668064504*pi) q[38];
U1q(0.60274237113598*pi,0.4576820823071399*pi) q[39];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[4],q[29];
RZZ(0.5*pi) q[5],q[32];
RZZ(0.5*pi) q[30],q[6];
RZZ(0.5*pi) q[21],q[7];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[36],q[14];
RZZ(0.5*pi) q[27],q[16];
RZZ(0.5*pi) q[17],q[35];
RZZ(0.5*pi) q[37],q[18];
RZZ(0.5*pi) q[31],q[19];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[28],q[24];
U1q(0.502751740187626*pi,0.44720527742005967*pi) q[0];
U1q(0.346640587614641*pi,1.1008769390799698*pi) q[1];
U1q(0.29453624303656*pi,0.74372682606508*pi) q[2];
U1q(0.226018261634371*pi,0.4863588115887003*pi) q[3];
U1q(0.77685347684642*pi,1.08857995144135*pi) q[4];
U1q(0.575342756844402*pi,0.3002676207053998*pi) q[5];
U1q(0.68004729271213*pi,1.7659967550095406*pi) q[6];
U1q(0.694952864190096*pi,1.4715528356357703*pi) q[7];
U1q(0.136888001482848*pi,1.0096103688809404*pi) q[8];
U1q(0.470606199613778*pi,0.5651882118074596*pi) q[9];
U1q(0.317003353613864*pi,0.7710789186424503*pi) q[10];
U1q(0.605855584641536*pi,1.35182082988615*pi) q[11];
U1q(0.779851515687927*pi,1.5508183719974902*pi) q[12];
U1q(0.365946063158257*pi,0.64322758783773*pi) q[13];
U1q(0.309214576145495*pi,1.0708916922072804*pi) q[14];
U1q(0.582869474065279*pi,1.4447948665353803*pi) q[15];
U1q(0.769714621179546*pi,0.4661343978302197*pi) q[16];
U1q(0.503836666124465*pi,1.0577196196469698*pi) q[17];
U1q(0.592584513235241*pi,1.1335171578469403*pi) q[18];
U1q(0.579930200745476*pi,1.3499234643804696*pi) q[19];
U1q(0.297055686531703*pi,1.3511435734749302*pi) q[20];
U1q(0.0347047712029326*pi,1.1577781944151697*pi) q[21];
U1q(0.236302753466411*pi,0.24326333102298037*pi) q[22];
U1q(0.574567365056454*pi,0.5972680638247398*pi) q[23];
U1q(0.644840472856025*pi,0.5702464034605903*pi) q[24];
U1q(0.361790120410085*pi,1.1773799613996392*pi) q[25];
U1q(0.649458672485589*pi,0.5591942089638202*pi) q[26];
U1q(0.643924201870675*pi,0.47808001174493064*pi) q[27];
U1q(0.67836338885138*pi,0.045759437557889804*pi) q[28];
U1q(0.802173078996572*pi,0.24549181483156968*pi) q[29];
U1q(0.199877743936845*pi,1.9769101412000802*pi) q[30];
U1q(0.201122672042636*pi,0.034106549082800086*pi) q[31];
U1q(0.203575284245069*pi,0.38027462571314974*pi) q[32];
U1q(0.307029736480589*pi,1.7863027408321592*pi) q[33];
U1q(0.0686788175173808*pi,1.9327741091566004*pi) q[34];
U1q(0.220789374617041*pi,0.8551769474825397*pi) q[35];
U1q(0.92319466231811*pi,0.4289047050025996*pi) q[36];
U1q(0.11672745654801*pi,1.5067376927295708*pi) q[37];
U1q(0.205964096879851*pi,0.2822169717292402*pi) q[38];
U1q(0.631396005173732*pi,1.6251956226755997*pi) q[39];
RZZ(0.5*pi) q[0],q[24];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[2],q[23];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[36],q[5];
RZZ(0.5*pi) q[38],q[6];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[32],q[8];
RZZ(0.5*pi) q[31],q[9];
RZZ(0.5*pi) q[35],q[11];
RZZ(0.5*pi) q[12],q[19];
RZZ(0.5*pi) q[21],q[13];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[37],q[16];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[27],q[22];
RZZ(0.5*pi) q[33],q[25];
RZZ(0.5*pi) q[34],q[26];
RZZ(0.5*pi) q[30],q[29];
U1q(0.799067283304646*pi,1.8129343664552504*pi) q[0];
U1q(0.315702538535889*pi,0.35925568596197*pi) q[1];
U1q(0.741334668779417*pi,0.8388745152073502*pi) q[2];
U1q(0.818134178023722*pi,1.8974109819455993*pi) q[3];
U1q(0.852021219870774*pi,1.4138519018029*pi) q[4];
U1q(0.526909425421077*pi,0.9507034762237492*pi) q[5];
U1q(0.153253571789971*pi,0.6533366769716995*pi) q[6];
U1q(0.762708146869415*pi,0.027146907763970773*pi) q[7];
U1q(0.261236905681824*pi,1.9668763223970007*pi) q[8];
U1q(0.576618694588932*pi,1.43539709107168*pi) q[9];
U1q(0.777435644563257*pi,1.6903797703438492*pi) q[10];
U1q(0.550832058979691*pi,1.2128707937972196*pi) q[11];
U1q(0.458545495620861*pi,1.0704388957723694*pi) q[12];
U1q(0.196544059092579*pi,1.3521988179615292*pi) q[13];
U1q(0.600848211746714*pi,1.2630556282052403*pi) q[14];
U1q(0.623775311351625*pi,1.3676245182521107*pi) q[15];
U1q(0.504854470796679*pi,0.5268776251345502*pi) q[16];
U1q(0.150414503324792*pi,1.1809463187723903*pi) q[17];
U1q(0.288695933491453*pi,1.17799281067866*pi) q[18];
U1q(0.452161169375292*pi,1.8822086587028704*pi) q[19];
U1q(0.609526446054419*pi,0.1908802459541299*pi) q[20];
U1q(0.777869104456182*pi,1.1140844530209204*pi) q[21];
U1q(0.394019770437508*pi,1.5088444980216007*pi) q[22];
U1q(0.318443481926709*pi,0.31087168603488013*pi) q[23];
U1q(0.49361851503517*pi,1.3474435197757*pi) q[24];
U1q(0.363567718303059*pi,0.3890761068127002*pi) q[25];
U1q(0.727467593614056*pi,0.38135993816709934*pi) q[26];
U1q(0.46497985528472*pi,1.9960960602114994*pi) q[27];
U1q(0.795423601394032*pi,0.44608725769604973*pi) q[28];
U1q(0.342804200733215*pi,0.19175515317950964*pi) q[29];
U1q(0.499501182379747*pi,1.8732389824313103*pi) q[30];
U1q(0.753399668918809*pi,0.5465882811827996*pi) q[31];
U1q(0.202576208996893*pi,1.8995940959571005*pi) q[32];
U1q(0.11815180702924*pi,0.17365706507839995*pi) q[33];
U1q(0.450377090519313*pi,1.1534791110266998*pi) q[34];
U1q(0.561062599681154*pi,0.03349645951550961*pi) q[35];
U1q(0.296992245955323*pi,0.5170567989634396*pi) q[36];
U1q(0.51795032796811*pi,0.9030682303605992*pi) q[37];
U1q(0.382726135464513*pi,1.0079092067608002*pi) q[38];
U1q(0.609794772133963*pi,1.0647356038636993*pi) q[39];
RZZ(0.5*pi) q[38],q[0];
RZZ(0.5*pi) q[17],q[1];
RZZ(0.5*pi) q[2],q[32];
RZZ(0.5*pi) q[29],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[5],q[35];
RZZ(0.5*pi) q[14],q[7];
RZZ(0.5*pi) q[19],q[8];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[12],q[28];
RZZ(0.5*pi) q[15],q[13];
RZZ(0.5*pi) q[26],q[16];
RZZ(0.5*pi) q[30],q[18];
RZZ(0.5*pi) q[21],q[25];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[33],q[36];
RZZ(0.5*pi) q[34],q[37];
U1q(0.316602735951372*pi,1.2730056318492*pi) q[0];
U1q(0.508007980440273*pi,1.5174588521954*pi) q[1];
U1q(0.158765731044127*pi,1.4195056065485492*pi) q[2];
U1q(0.506050489400999*pi,0.08307084016620081*pi) q[3];
U1q(0.35593109826619*pi,0.05628099414060017*pi) q[4];
U1q(0.283411425918029*pi,1.8432409356805994*pi) q[5];
U1q(0.731532661459777*pi,0.7701701371313003*pi) q[6];
U1q(0.420549704258785*pi,0.5113176632302991*pi) q[7];
U1q(0.51314043707097*pi,1.8178800631605991*pi) q[8];
U1q(0.238103782092668*pi,1.1718154881205205*pi) q[9];
U1q(0.849091137118474*pi,0.09407670111098021*pi) q[10];
U1q(0.559253159846394*pi,0.07969545023413005*pi) q[11];
U1q(0.603230626262671*pi,1.2937131584761001*pi) q[12];
U1q(0.249853929299935*pi,1.2794251665303005*pi) q[13];
U1q(0.456515565884486*pi,0.6388242272747*pi) q[14];
U1q(0.62629433965162*pi,1.1457918274168009*pi) q[15];
U1q(0.547353663836309*pi,0.5703828355914897*pi) q[16];
U1q(0.645646572080495*pi,1.4404259402624007*pi) q[17];
U1q(0.411546455019734*pi,0.9561424051154006*pi) q[18];
U1q(0.568156408610961*pi,0.35389149966700995*pi) q[19];
U1q(0.762204228044986*pi,1.9586604913376995*pi) q[20];
U1q(0.281424186986735*pi,0.7858208256325998*pi) q[21];
U1q(0.913501328522088*pi,1.2865753677524996*pi) q[22];
U1q(0.764172362007157*pi,0.9781036069929208*pi) q[23];
U1q(0.50394288962926*pi,0.13099376877593016*pi) q[24];
U1q(0.464256862946729*pi,0.26794983900859926*pi) q[25];
U1q(0.842839606764488*pi,1.4508041201983009*pi) q[26];
U1q(0.520572983903166*pi,0.6327805892879006*pi) q[27];
U1q(0.397065537715427*pi,1.8565994295187007*pi) q[28];
U1q(0.775836423259505*pi,1.6882571202701993*pi) q[29];
U1q(0.670352977425154*pi,1.2934837905708996*pi) q[30];
U1q(0.251855958162886*pi,1.5124830430491691*pi) q[31];
U1q(0.76098328635631*pi,0.1289459452216999*pi) q[32];
U1q(0.20795466905141*pi,0.2488576357658996*pi) q[33];
U1q(0.57851740282071*pi,1.5687963361280008*pi) q[34];
U1q(0.578575452372816*pi,1.2211139570099991*pi) q[35];
U1q(0.392441010078476*pi,0.16198932187620052*pi) q[36];
U1q(0.595926545449273*pi,1.2961699196828*pi) q[37];
U1q(0.886993011876476*pi,0.6645590313043996*pi) q[38];
U1q(0.0416818308170675*pi,1.5290869754504008*pi) q[39];
RZZ(0.5*pi) q[2],q[0];
RZZ(0.5*pi) q[39],q[1];
RZZ(0.5*pi) q[25],q[3];
RZZ(0.5*pi) q[4],q[13];
RZZ(0.5*pi) q[5],q[17];
RZZ(0.5*pi) q[6],q[29];
RZZ(0.5*pi) q[36],q[7];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[26];
RZZ(0.5*pi) q[27],q[11];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[15],q[22];
RZZ(0.5*pi) q[16],q[35];
RZZ(0.5*pi) q[19],q[18];
RZZ(0.5*pi) q[20],q[32];
RZZ(0.5*pi) q[21],q[28];
RZZ(0.5*pi) q[23],q[24];
RZZ(0.5*pi) q[33],q[30];
U1q(0.854513344920764*pi,1.4509719002201003*pi) q[0];
U1q(0.299422933794289*pi,1.3014646935399998*pi) q[1];
U1q(0.455539368720118*pi,1.9997427864554993*pi) q[2];
U1q(0.600742769769601*pi,0.3549899616419996*pi) q[3];
U1q(0.546841824715376*pi,0.7476215771251002*pi) q[4];
U1q(0.165656326463641*pi,0.9240853246911005*pi) q[5];
U1q(0.297928451524022*pi,1.563584280634199*pi) q[6];
U1q(0.858977897683383*pi,1.0165538977581008*pi) q[7];
U1q(0.489421574030916*pi,0.15060950216929925*pi) q[8];
U1q(0.555888324483779*pi,0.33370537481270013*pi) q[9];
U1q(0.682719620663485*pi,0.15722633537350994*pi) q[10];
U1q(0.485566988153852*pi,0.2905503795096003*pi) q[11];
U1q(0.176208080547127*pi,1.3516302085247993*pi) q[12];
U1q(0.885625205236448*pi,1.7803203180160008*pi) q[13];
U1q(0.700169045518086*pi,1.4789126812218*pi) q[14];
U1q(0.829201444592333*pi,0.9328601846696003*pi) q[15];
U1q(0.668504399047239*pi,1.3150010192187*pi) q[16];
U1q(0.25955462160909*pi,0.8127750323283998*pi) q[17];
U1q(0.189351500897416*pi,1.6405425804048992*pi) q[18];
U1q(0.869925095228464*pi,0.8977601462922298*pi) q[19];
U1q(0.512410058889959*pi,1.9625994756446996*pi) q[20];
U1q(0.698799846774496*pi,1.7145613670657003*pi) q[21];
U1q(0.607409358021225*pi,1.8708549974198014*pi) q[22];
U1q(0.249201549165056*pi,0.6047685181210998*pi) q[23];
U1q(0.433366408016318*pi,1.0535721318377007*pi) q[24];
U1q(0.258635742150994*pi,1.848250589484799*pi) q[25];
U1q(0.910627006452994*pi,1.8923453328234991*pi) q[26];
U1q(0.346840757734513*pi,0.45112034764619935*pi) q[27];
U1q(0.390203992927551*pi,1.6039013005510991*pi) q[28];
U1q(0.513835962815401*pi,1.7205446125794008*pi) q[29];
U1q(0.722508873385378*pi,0.9900070983578999*pi) q[30];
U1q(0.0807749916231691*pi,1.5053146955059002*pi) q[31];
U1q(0.403476263278687*pi,1.990545255829801*pi) q[32];
U1q(0.259529073585085*pi,0.32978609934919945*pi) q[33];
U1q(0.391181387351353*pi,1.3667987269279003*pi) q[34];
U1q(0.649364513533154*pi,0.5592302790182995*pi) q[35];
U1q(0.447364146412069*pi,1.3421361138174*pi) q[36];
U1q(0.287022754075046*pi,1.4055705415836997*pi) q[37];
U1q(0.863543203552062*pi,1.6701178331351993*pi) q[38];
U1q(0.589391845681313*pi,1.6209396382744998*pi) q[39];
RZZ(0.5*pi) q[0],q[4];
RZZ(0.5*pi) q[5],q[1];
RZZ(0.5*pi) q[2],q[34];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[13],q[8];
RZZ(0.5*pi) q[37],q[9];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[12],q[26];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[27],q[18];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[22],q[24];
RZZ(0.5*pi) q[30],q[25];
RZZ(0.5*pi) q[39],q[28];
RZZ(0.5*pi) q[38],q[29];
RZZ(0.5*pi) q[33],q[31];
RZZ(0.5*pi) q[36],q[35];
U1q(0.42937832805898*pi,1.9081890015367016*pi) q[0];
U1q(0.444919039697579*pi,1.7330632496327993*pi) q[1];
U1q(0.935972864283022*pi,1.7630621026526008*pi) q[2];
U1q(0.564755935858569*pi,1.5127452241617014*pi) q[3];
U1q(0.608456822121863*pi,1.8537585369118013*pi) q[4];
U1q(0.62311101573745*pi,1.6597951224743*pi) q[5];
U1q(0.501535424867098*pi,1.6863552402944997*pi) q[6];
U1q(0.510827244323965*pi,1.3333203835934988*pi) q[7];
U1q(0.550321389321914*pi,0.23091059915159917*pi) q[8];
U1q(0.20770776350659*pi,0.03955000910550055*pi) q[9];
U1q(0.720794431777695*pi,1.1097209647975994*pi) q[10];
U1q(0.784272029806495*pi,0.9681775145462996*pi) q[11];
U1q(0.374385704698347*pi,1.7867611715447005*pi) q[12];
U1q(0.717111251712567*pi,0.6236022491229001*pi) q[13];
U1q(0.494544037139451*pi,1.4732053394322993*pi) q[14];
U1q(0.411682093262096*pi,0.015339968524600067*pi) q[15];
U1q(0.176710905468286*pi,1.761786029625*pi) q[16];
U1q(0.0677673612392772*pi,1.0504813851616994*pi) q[17];
U1q(0.156323022403663*pi,1.6943584320721996*pi) q[18];
U1q(0.409059966544728*pi,1.0183884514717008*pi) q[19];
U1q(0.207859417380594*pi,1.0970917795152992*pi) q[20];
U1q(0.518194683883931*pi,0.7924424970852009*pi) q[21];
U1q(0.504059866773406*pi,0.2075725111552984*pi) q[22];
U1q(0.754806753917447*pi,1.7385594011518997*pi) q[23];
U1q(0.392726820765585*pi,1.9471945309541994*pi) q[24];
U1q(0.743176723731761*pi,0.22182775599889837*pi) q[25];
U1q(0.636243711047301*pi,1.6993287541555002*pi) q[26];
U1q(0.309332555767333*pi,1.8779801852693012*pi) q[27];
U1q(0.530784849870922*pi,1.4823461513493008*pi) q[28];
U1q(0.723766409726246*pi,1.3573979553531998*pi) q[29];
U1q(0.574990628233712*pi,1.5485444513744007*pi) q[30];
U1q(0.493514416016453*pi,1.6158462931920994*pi) q[31];
U1q(0.95171483753888*pi,1.0416895951412997*pi) q[32];
U1q(0.466706554255344*pi,1.4264887400739994*pi) q[33];
U1q(0.651388647216493*pi,1.757870813075499*pi) q[34];
U1q(0.31794097076943*pi,0.3531537231881998*pi) q[35];
U1q(0.338513199474567*pi,1.6553919485800002*pi) q[36];
U1q(0.479554158791301*pi,0.024683343706200844*pi) q[37];
U1q(0.303553139811429*pi,0.35508802946839957*pi) q[38];
U1q(0.724319001464521*pi,0.9697992251755991*pi) q[39];
RZZ(0.5*pi) q[0],q[29];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[4],q[18];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[25],q[9];
RZZ(0.5*pi) q[36],q[10];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[14],q[13];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[34],q[17];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[21],q[26];
RZZ(0.5*pi) q[28],q[23];
RZZ(0.5*pi) q[31],q[35];
U1q(0.684723506136536*pi,0.7951426608446006*pi) q[0];
U1q(0.329018166930156*pi,1.5546782377764998*pi) q[1];
U1q(0.59360400479241*pi,0.7638939031120007*pi) q[2];
U1q(0.563989251776063*pi,1.5395215563316995*pi) q[3];
U1q(0.470525284849761*pi,0.8367557853751002*pi) q[4];
U1q(0.701461489082163*pi,1.5618596243715004*pi) q[5];
U1q(0.187415092240822*pi,1.0076615736309016*pi) q[6];
U1q(0.265045408105617*pi,0.30422028809369905*pi) q[7];
U1q(0.807853351663655*pi,0.4260396544972984*pi) q[8];
U1q(0.575290120773917*pi,1.3641408735017997*pi) q[9];
U1q(0.363722381563676*pi,1.3405312588590004*pi) q[10];
U1q(0.381936642358521*pi,1.0014221903584009*pi) q[11];
U1q(0.245612082857873*pi,1.4250136731412013*pi) q[12];
U1q(0.462560453259506*pi,0.38919379000179966*pi) q[13];
U1q(0.57515437136049*pi,0.3609760065610992*pi) q[14];
U1q(0.552511348683814*pi,0.534435697629899*pi) q[15];
U1q(0.831622201986589*pi,0.8102665048463003*pi) q[16];
U1q(0.513313566982431*pi,1.3254366041193997*pi) q[17];
U1q(0.95068442041024*pi,0.04436051859610046*pi) q[18];
U1q(0.700253043876449*pi,0.5928153077146998*pi) q[19];
U1q(0.388030773196344*pi,0.35097917693190084*pi) q[20];
U1q(0.578926721018529*pi,1.038896927109299*pi) q[21];
U1q(0.383874597768869*pi,1.7335858218864004*pi) q[22];
U1q(0.633252886423151*pi,1.4414110187465*pi) q[23];
U1q(0.742586234359899*pi,0.021312480538099976*pi) q[24];
U1q(0.61637640072575*pi,1.5239618783308018*pi) q[25];
U1q(0.529594379504878*pi,1.8984840873761009*pi) q[26];
U1q(0.844078224881144*pi,1.9378172543976007*pi) q[27];
U1q(0.441111242238675*pi,0.2631977010673996*pi) q[28];
U1q(0.531656642319942*pi,1.0478266681598*pi) q[29];
U1q(0.266127542402083*pi,1.1538535279592992*pi) q[30];
U1q(0.390777892044088*pi,0.9448820871273007*pi) q[31];
U1q(0.349304228357805*pi,0.5457192227952987*pi) q[32];
U1q(0.183897115700132*pi,1.1177928596655988*pi) q[33];
U1q(0.324359313034487*pi,1.1999485123381994*pi) q[34];
U1q(0.612975124114053*pi,0.850953271761199*pi) q[35];
U1q(0.369036588212463*pi,0.0767907063146005*pi) q[36];
U1q(0.318329733535407*pi,0.8616708620793005*pi) q[37];
U1q(0.367938892137677*pi,1.085758056080099*pi) q[38];
U1q(0.809875865100307*pi,1.2963219833242015*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[2],q[13];
RZZ(0.5*pi) q[19],q[3];
RZZ(0.5*pi) q[15],q[4];
RZZ(0.5*pi) q[5],q[26];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[39],q[7];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[18],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[36],q[11];
RZZ(0.5*pi) q[12],q[16];
RZZ(0.5*pi) q[17],q[31];
RZZ(0.5*pi) q[20],q[30];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[38],q[27];
RZZ(0.5*pi) q[28],q[32];
U1q(0.389350327470456*pi,1.7130543928141009*pi) q[0];
U1q(0.806224783368921*pi,1.9445672001580014*pi) q[1];
U1q(0.340169552968285*pi,0.3933807667393996*pi) q[2];
U1q(0.566161790360029*pi,0.30587236921919825*pi) q[3];
U1q(0.268731866626494*pi,0.5087554458779984*pi) q[4];
U1q(0.426249115830243*pi,1.1174255456069986*pi) q[5];
U1q(0.427867884091231*pi,1.4683638945923008*pi) q[6];
U1q(0.375005931659324*pi,1.8924004989642*pi) q[7];
U1q(0.213907306524084*pi,0.7520055865666002*pi) q[8];
U1q(0.634479202785951*pi,1.3760038343749983*pi) q[9];
U1q(0.793221613466141*pi,1.5501605415624997*pi) q[10];
U1q(0.50247100793401*pi,1.8632897262054016*pi) q[11];
U1q(0.707999410231115*pi,0.4020357528980014*pi) q[12];
U1q(0.728544227598242*pi,1.4784104167847012*pi) q[13];
U1q(0.328150500205671*pi,0.011437515453998515*pi) q[14];
U1q(0.571747082403214*pi,1.1224913519990984*pi) q[15];
U1q(0.785231358639471*pi,1.9806896007894998*pi) q[16];
U1q(0.791260942357301*pi,1.7647644549971986*pi) q[17];
U1q(0.770543142852969*pi,0.4146929305427989*pi) q[18];
U1q(0.148200319390414*pi,1.9647195656511016*pi) q[19];
U1q(0.649791579439727*pi,1.204471634042001*pi) q[20];
U1q(0.704824681961345*pi,0.5524628981797015*pi) q[21];
U1q(0.435247324061543*pi,0.07163267759910141*pi) q[22];
U1q(0.333286108843071*pi,1.915298142659001*pi) q[23];
U1q(0.15679158284543*pi,1.2103224195454985*pi) q[24];
U1q(0.824946568341511*pi,0.03746906000349881*pi) q[25];
U1q(0.169455671734796*pi,1.1065613804531012*pi) q[26];
U1q(0.234019958555437*pi,1.1107659164909016*pi) q[27];
U1q(0.324534721347308*pi,0.48184065061100156*pi) q[28];
U1q(0.55022881679971*pi,0.9933521192808001*pi) q[29];
U1q(0.277248142541506*pi,0.9195928713271009*pi) q[30];
U1q(0.636255270931208*pi,1.9162255267901003*pi) q[31];
U1q(0.61765473059575*pi,0.9982407673828995*pi) q[32];
U1q(0.111874632141713*pi,0.9826719982491987*pi) q[33];
U1q(0.685124121916612*pi,0.7948446056491001*pi) q[34];
U1q(0.882116659010387*pi,0.016866446193301243*pi) q[35];
U1q(0.617458097637987*pi,1.0320594692815988*pi) q[36];
U1q(0.59827381190766*pi,0.5318647995387984*pi) q[37];
U1q(0.356771857091751*pi,0.9843138391529997*pi) q[38];
U1q(0.683252973252226*pi,1.2854595123578996*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[18];
RZZ(0.5*pi) q[2],q[35];
RZZ(0.5*pi) q[34],q[3];
RZZ(0.5*pi) q[17],q[4];
RZZ(0.5*pi) q[5],q[29];
RZZ(0.5*pi) q[19],q[6];
RZZ(0.5*pi) q[22],q[7];
RZZ(0.5*pi) q[33],q[8];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[10],q[31];
RZZ(0.5*pi) q[38],q[12];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[20],q[25];
RZZ(0.5*pi) q[21],q[36];
RZZ(0.5*pi) q[27],q[24];
RZZ(0.5*pi) q[37],q[28];
U1q(0.422796569799217*pi,1.6919818888198002*pi) q[0];
U1q(0.829361064365943*pi,0.31558721396849876*pi) q[1];
U1q(0.550800672516807*pi,0.5444301623256003*pi) q[2];
U1q(0.200595207924746*pi,0.7942886458769998*pi) q[3];
U1q(0.15991900195894*pi,0.7814843820214001*pi) q[4];
U1q(0.639082861055988*pi,1.3720435457040985*pi) q[5];
U1q(0.281686042073605*pi,0.25800572998150173*pi) q[6];
U1q(0.466877022946385*pi,1.2707297812051017*pi) q[7];
U1q(0.612889392707034*pi,1.5768185710794995*pi) q[8];
U1q(0.746487822248249*pi,1.8855793244064003*pi) q[9];
U1q(0.566456798471091*pi,1.5216984498110016*pi) q[10];
U1q(0.406740469441252*pi,1.763752031685101*pi) q[11];
U1q(0.732717199902293*pi,0.31304242155700024*pi) q[12];
U1q(0.674425376288034*pi,1.3551656241333987*pi) q[13];
U1q(0.457501297313439*pi,0.8203724991609*pi) q[14];
U1q(0.443032236499443*pi,1.5598792270916988*pi) q[15];
U1q(0.411702952088793*pi,0.8505708096067011*pi) q[16];
U1q(0.476523699737075*pi,1.0150499008762992*pi) q[17];
U1q(0.322293818785859*pi,0.047896806076700216*pi) q[18];
U1q(0.469560332874421*pi,1.4997179924751016*pi) q[19];
U1q(0.299654524456011*pi,1.4923381379208003*pi) q[20];
U1q(0.680960483994601*pi,0.7613538676665996*pi) q[21];
U1q(0.604935655001104*pi,0.7703807073373987*pi) q[22];
U1q(0.450795274615844*pi,0.8968793768761998*pi) q[23];
U1q(0.754823911965998*pi,0.3265694290025998*pi) q[24];
U1q(0.888772649990505*pi,0.04798133800949955*pi) q[25];
U1q(0.904039320766518*pi,0.5325994693911014*pi) q[26];
U1q(0.14421307641135*pi,1.7496402196305993*pi) q[27];
U1q(0.450460439232049*pi,0.9168756845459001*pi) q[28];
U1q(0.309602040828159*pi,1.564385140710801*pi) q[29];
U1q(0.490792993599228*pi,1.5754648356407017*pi) q[30];
U1q(0.51052771709756*pi,0.2481298979744011*pi) q[31];
U1q(0.676875912485579*pi,1.6939819652012993*pi) q[32];
U1q(0.439377606690824*pi,1.024043536379299*pi) q[33];
U1q(0.902661269497568*pi,0.11058710968750063*pi) q[34];
U1q(0.302311889958686*pi,0.2882970544919985*pi) q[35];
U1q(0.860429219138136*pi,0.10506590399389992*pi) q[36];
U1q(0.514381361062373*pi,1.2822385050830007*pi) q[37];
U1q(0.841316418670811*pi,0.2728072150496992*pi) q[38];
U1q(0.202690810357353*pi,0.956766246562399*pi) q[39];
RZZ(0.5*pi) q[0],q[8];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[2],q[5];
RZZ(0.5*pi) q[37],q[3];
RZZ(0.5*pi) q[4],q[16];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[13],q[7];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[12],q[27];
RZZ(0.5*pi) q[14],q[30];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[36],q[19];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[21],q[24];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[34],q[25];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[33],q[29];
U1q(0.76557663712021*pi,1.7377567697612015*pi) q[0];
U1q(0.284804841113181*pi,0.5243010551801994*pi) q[1];
U1q(0.59601117519387*pi,1.4779390698869008*pi) q[2];
U1q(0.0955522914159208*pi,1.7140479645622015*pi) q[3];
U1q(0.391616043663668*pi,0.7463513243085984*pi) q[4];
U1q(0.107276668603803*pi,0.8053250247963*pi) q[5];
U1q(0.585603615414484*pi,1.8182153971287*pi) q[6];
U1q(0.429714357921447*pi,1.1401556715986985*pi) q[7];
U1q(0.5833305644043*pi,0.2729450777613991*pi) q[8];
U1q(0.224095103030085*pi,1.6466805508967006*pi) q[9];
U1q(0.726011603761743*pi,1.1602721834429985*pi) q[10];
U1q(0.427860902297493*pi,1.1735603242162007*pi) q[11];
U1q(0.489754098478274*pi,0.027571428120001684*pi) q[12];
U1q(0.82757049324953*pi,1.7749687755889987*pi) q[13];
U1q(0.778751055332009*pi,1.1393267486477008*pi) q[14];
U1q(0.308373018788413*pi,1.8868013030743*pi) q[15];
U1q(0.649991343874195*pi,1.5265399911313011*pi) q[16];
U1q(0.582903260911563*pi,0.15973481576349968*pi) q[17];
U1q(0.468106778324639*pi,0.43371657543400133*pi) q[18];
U1q(0.64361517263243*pi,1.1874200830886998*pi) q[19];
U1q(0.758765020251418*pi,0.5648819166008998*pi) q[20];
U1q(0.467186711084279*pi,1.7058452353513012*pi) q[21];
U1q(0.719982147541247*pi,0.26492409784129833*pi) q[22];
U1q(0.182757166363817*pi,1.3201884486318995*pi) q[23];
U1q(0.768688102391596*pi,0.17512465699169866*pi) q[24];
U1q(0.745370234118342*pi,0.9656652036020006*pi) q[25];
U1q(0.530274300254981*pi,0.8881046575097002*pi) q[26];
U1q(0.463814893007206*pi,1.411902080328499*pi) q[27];
U1q(0.606675624488236*pi,0.9199096151317008*pi) q[28];
U1q(0.536586317183689*pi,1.7667456003399984*pi) q[29];
U1q(0.770222222412441*pi,1.7264379114504003*pi) q[30];
U1q(0.248752656997306*pi,0.1324645002449003*pi) q[31];
U1q(0.437056583181114*pi,1.9170787354764016*pi) q[32];
U1q(0.579912207507795*pi,0.1704912443921991*pi) q[33];
U1q(0.442526150535176*pi,0.7710224556148013*pi) q[34];
U1q(0.597738249975936*pi,0.22559468088460122*pi) q[35];
U1q(0.479957596555368*pi,1.5244788770122*pi) q[36];
U1q(0.382729075537314*pi,1.1819454073565012*pi) q[37];
U1q(0.538183922547452*pi,0.9071131749258008*pi) q[38];
U1q(0.432249040636549*pi,0.645940022935001*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[29];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[4],q[19];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[21],q[6];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[12],q[11];
RZZ(0.5*pi) q[36],q[13];
RZZ(0.5*pi) q[14],q[25];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[16],q[18];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[39],q[23];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[38],q[26];
RZZ(0.5*pi) q[30],q[28];
RZZ(0.5*pi) q[31],q[32];
U1q(0.613303026923161*pi,1.2724768602911993*pi) q[0];
U1q(0.444249680882318*pi,0.8942371354022995*pi) q[1];
U1q(0.410355967756547*pi,1.393229873881701*pi) q[2];
U1q(0.0380158087630868*pi,0.4766580570911998*pi) q[3];
U1q(0.386671212823169*pi,0.12780963745339946*pi) q[4];
U1q(0.338338250244072*pi,0.5099490375357014*pi) q[5];
U1q(0.512702293196447*pi,0.5894148443371989*pi) q[6];
U1q(0.722680646824869*pi,0.5013918288827988*pi) q[7];
U1q(0.0873572998839973*pi,0.4142976014741002*pi) q[8];
U1q(0.549363529640248*pi,0.3081904408706997*pi) q[9];
U1q(0.875330239077272*pi,0.888665183006399*pi) q[10];
U1q(0.385479059351075*pi,1.6509791534185005*pi) q[11];
U1q(0.685918386309984*pi,1.604876312284599*pi) q[12];
U1q(0.294722205817268*pi,0.849313378215701*pi) q[13];
U1q(0.713658689832688*pi,0.05351266118839959*pi) q[14];
U1q(0.531730702799451*pi,1.6629958810957994*pi) q[15];
U1q(0.56290735898173*pi,1.8074668129379994*pi) q[16];
U1q(0.287024110843491*pi,0.5634135432731995*pi) q[17];
U1q(0.701532208886828*pi,0.3266904949211984*pi) q[18];
U1q(0.644993784311563*pi,1.4823571473394992*pi) q[19];
U1q(0.386983689027728*pi,1.9976413499983003*pi) q[20];
U1q(0.434262788308999*pi,1.0192745608179017*pi) q[21];
U1q(0.785548625214234*pi,1.4569287206622974*pi) q[22];
U1q(0.496159843747312*pi,1.3252384310588994*pi) q[23];
U1q(0.63389698665385*pi,1.3736664400469998*pi) q[24];
U1q(0.362796009923823*pi,1.7325778717087985*pi) q[25];
U1q(0.452979525872643*pi,1.9937225568984012*pi) q[26];
U1q(0.758327331788652*pi,1.3917054774549982*pi) q[27];
U1q(0.804615775528387*pi,0.805540903792501*pi) q[28];
U1q(0.655440672667763*pi,0.5500851984497004*pi) q[29];
U1q(0.842817072016692*pi,1.3124851042713992*pi) q[30];
U1q(0.720711696217443*pi,0.7537544147603015*pi) q[31];
U1q(0.921458355410158*pi,0.4686535729576988*pi) q[32];
U1q(0.137325090343357*pi,1.8899139459634*pi) q[33];
U1q(0.378350520147843*pi,1.553226867454299*pi) q[34];
U1q(0.476500906508838*pi,1.6203258616172995*pi) q[35];
U1q(0.312992209762727*pi,0.45452840326669985*pi) q[36];
U1q(0.425892279062657*pi,0.09924159889429873*pi) q[37];
U1q(0.920059368795709*pi,0.3400182742438993*pi) q[38];
U1q(0.412257073142284*pi,0.7366974619309019*pi) q[39];
rz(2.4735145315953986*pi) q[0];
rz(2.107321545277401*pi) q[1];
rz(3.2872837701177*pi) q[2];
rz(3.4739272025235017*pi) q[3];
rz(1.0976294249894991*pi) q[4];
rz(0.9500657196870996*pi) q[5];
rz(0.04296293974169885*pi) q[6];
rz(1.7421590208764002*pi) q[7];
rz(3.1562649241608014*pi) q[8];
rz(0.6907306072173007*pi) q[9];
rz(3.8893390429430994*pi) q[10];
rz(0.42382183198590084*pi) q[11];
rz(2.0001112204776987*pi) q[12];
rz(2.8516057272656*pi) q[13];
rz(0.9566223432405998*pi) q[14];
rz(2.461701017229899*pi) q[15];
rz(1.9135089573138018*pi) q[16];
rz(3.858539285125701*pi) q[17];
rz(2.739905688042299*pi) q[18];
rz(2.5831659864503003*pi) q[19];
rz(2.5244217824567983*pi) q[20];
rz(1.1043660732449005*pi) q[21];
rz(2.275357231926698*pi) q[22];
rz(3.6004339840688004*pi) q[23];
rz(3.6526487910858982*pi) q[24];
rz(2.0133327585893*pi) q[25];
rz(2.8789462070265017*pi) q[26];
rz(0.400600446360599*pi) q[27];
rz(3.131956565264801*pi) q[28];
rz(1.5835685248421*pi) q[29];
rz(1.8738707126901986*pi) q[30];
rz(3.2054941031739013*pi) q[31];
rz(1.8041012420002005*pi) q[32];
rz(0.35682781244440065*pi) q[33];
rz(3.938446543754001*pi) q[34];
rz(1.532747836081601*pi) q[35];
rz(3.3063345317373987*pi) q[36];
rz(2.614918662179999*pi) q[37];
rz(2.3245917530278*pi) q[38];
rz(1.3220639134968977*pi) q[39];
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