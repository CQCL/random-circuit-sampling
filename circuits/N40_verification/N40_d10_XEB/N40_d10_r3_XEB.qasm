OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.714619960762805*pi,0.0796952408989157*pi) q[0];
U1q(0.10871318763177*pi,1.124780807948281*pi) q[1];
U1q(0.38036593308456*pi,0.60920442684483*pi) q[2];
U1q(0.235292263847702*pi,0.69020381097267*pi) q[3];
U1q(0.606191248064538*pi,1.737473041935711*pi) q[4];
U1q(0.758993165518166*pi,1.809558884450229*pi) q[5];
U1q(0.714066868420827*pi,1.495533677002886*pi) q[6];
U1q(0.50142907467598*pi,1.42055992926494*pi) q[7];
U1q(0.538518779714627*pi,0.278975045590488*pi) q[8];
U1q(0.505931755961052*pi,1.853958896462192*pi) q[9];
U1q(0.174274571888099*pi,1.587377416559749*pi) q[10];
U1q(0.0947600814567086*pi,1.812548542681489*pi) q[11];
U1q(0.617831575181056*pi,1.5308038149781371*pi) q[12];
U1q(0.468957414804382*pi,1.821330617628653*pi) q[13];
U1q(0.315732452679979*pi,0.256293944692672*pi) q[14];
U1q(0.807670600403968*pi,0.426373891853452*pi) q[15];
U1q(0.252818432589741*pi,0.100358456623058*pi) q[16];
U1q(0.773578059281735*pi,0.266687178278847*pi) q[17];
U1q(0.249942530795283*pi,0.287144827373795*pi) q[18];
U1q(0.255973420053714*pi,1.05060311735944*pi) q[19];
U1q(0.428344930889525*pi,0.69612298154011*pi) q[20];
U1q(0.970370907722135*pi,1.638946206993503*pi) q[21];
U1q(0.403758295438356*pi,0.6712994573858699*pi) q[22];
U1q(0.614590173186336*pi,1.08224094472457*pi) q[23];
U1q(0.677254347215364*pi,1.87459363538901*pi) q[24];
U1q(0.638724867374283*pi,1.626052087814765*pi) q[25];
U1q(0.715875132009888*pi,1.79036488920037*pi) q[26];
U1q(0.845850733992764*pi,1.121318130838102*pi) q[27];
U1q(0.464280199849793*pi,1.471003401956765*pi) q[28];
U1q(0.608846964458052*pi,0.904523653152597*pi) q[29];
U1q(0.463623772786072*pi,0.170562334267241*pi) q[30];
U1q(0.73628920949398*pi,0.603637289814331*pi) q[31];
U1q(0.569954231716836*pi,1.40639585328802*pi) q[32];
U1q(0.675041813489024*pi,0.74035678739635*pi) q[33];
U1q(0.472761630315093*pi,1.570637594024815*pi) q[34];
U1q(0.528498185837673*pi,0.236631225920035*pi) q[35];
U1q(0.557904646641426*pi,0.637909921470468*pi) q[36];
U1q(0.154981390141702*pi,1.799868787601367*pi) q[37];
U1q(0.358632835442978*pi,1.9404815196437586*pi) q[38];
U1q(0.274989708569911*pi,0.833510798269232*pi) q[39];
RZZ(0.5*pi) q[29],q[0];
RZZ(0.5*pi) q[1],q[26];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[3],q[16];
RZZ(0.5*pi) q[4],q[21];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[24],q[11];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[18],q[34];
RZZ(0.5*pi) q[28],q[19];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[35],q[23];
RZZ(0.5*pi) q[36],q[30];
RZZ(0.5*pi) q[31],q[37];
U1q(0.595843946723834*pi,1.82312059403369*pi) q[0];
U1q(0.497434935248881*pi,0.09231498606240995*pi) q[1];
U1q(0.53693435486647*pi,0.5380317527345202*pi) q[2];
U1q(0.15962797269784*pi,0.7890505885195598*pi) q[3];
U1q(0.366312097943896*pi,1.11761537850577*pi) q[4];
U1q(0.595293958957193*pi,0.8755404050062499*pi) q[5];
U1q(0.404912569081125*pi,0.8538878277072199*pi) q[6];
U1q(0.606253742084375*pi,1.389477199115524*pi) q[7];
U1q(0.621936605341962*pi,0.15670889098811003*pi) q[8];
U1q(0.743040474105236*pi,0.89633886945842*pi) q[9];
U1q(0.379925790554427*pi,0.33583189842042005*pi) q[10];
U1q(0.649204187111835*pi,0.7254796518302*pi) q[11];
U1q(0.323797138029347*pi,0.98271423712574*pi) q[12];
U1q(0.156075758718755*pi,0.42848198649816993*pi) q[13];
U1q(0.752688936113061*pi,0.21504537185254002*pi) q[14];
U1q(0.170633578570746*pi,0.33230438296065*pi) q[15];
U1q(0.198344492679958*pi,1.39893314698178*pi) q[16];
U1q(0.15134670262197*pi,0.92706869879401*pi) q[17];
U1q(0.563984381719632*pi,0.64453331396892*pi) q[18];
U1q(0.760466019714597*pi,1.5240411685893802*pi) q[19];
U1q(0.965934967536642*pi,1.92250754669094*pi) q[20];
U1q(0.631347949849764*pi,1.84533839024551*pi) q[21];
U1q(0.670220812263153*pi,1.22338242599777*pi) q[22];
U1q(0.381195758007393*pi,1.056124859587828*pi) q[23];
U1q(0.570052581407605*pi,0.41873417097787*pi) q[24];
U1q(0.811464640296155*pi,0.6452215659895102*pi) q[25];
U1q(0.363034677960149*pi,1.5320357205439001*pi) q[26];
U1q(0.551003185962341*pi,1.12418104525905*pi) q[27];
U1q(0.444400046962153*pi,1.18634526721811*pi) q[28];
U1q(0.396134207269157*pi,1.2102328802778919*pi) q[29];
U1q(0.489866208118888*pi,0.04885619551029996*pi) q[30];
U1q(0.563555211148676*pi,0.25656927768749993*pi) q[31];
U1q(0.124658549921981*pi,0.211776815803449*pi) q[32];
U1q(0.59797048079781*pi,1.1613467773707051*pi) q[33];
U1q(0.588513488612239*pi,0.6708818603981599*pi) q[34];
U1q(0.615519961103839*pi,1.3148158309814102*pi) q[35];
U1q(0.927791465816784*pi,1.8431737413317801*pi) q[36];
U1q(0.22354286296781*pi,0.048969124470539915*pi) q[37];
U1q(0.618910857232114*pi,1.9529033924341257*pi) q[38];
U1q(0.575888922043298*pi,1.152820078698743*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[2],q[19];
RZZ(0.5*pi) q[3],q[14];
RZZ(0.5*pi) q[4],q[38];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[36],q[7];
RZZ(0.5*pi) q[8],q[28];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[37],q[16];
RZZ(0.5*pi) q[17],q[25];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[22],q[23];
RZZ(0.5*pi) q[33],q[24];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[30],q[27];
RZZ(0.5*pi) q[29],q[32];
U1q(0.566280033749663*pi,0.7700139734157299*pi) q[0];
U1q(0.308454865627487*pi,0.98261976303335*pi) q[1];
U1q(0.729291922776299*pi,0.7278881136909003*pi) q[2];
U1q(0.0317511585743189*pi,0.6518468421161998*pi) q[3];
U1q(0.214661915314537*pi,1.651277287898*pi) q[4];
U1q(0.258491669238953*pi,1.9610100441749498*pi) q[5];
U1q(0.309290838291195*pi,1.4999573773265604*pi) q[6];
U1q(0.417587621431103*pi,0.46769668006689*pi) q[7];
U1q(0.299933890787889*pi,0.7377530171375604*pi) q[8];
U1q(0.208883304908754*pi,1.8104890364983603*pi) q[9];
U1q(0.361375295535982*pi,1.99003181845986*pi) q[10];
U1q(0.561196629462504*pi,1.87788618188458*pi) q[11];
U1q(0.613172393955156*pi,0.8141241847000202*pi) q[12];
U1q(0.791451113485843*pi,1.8507278818699504*pi) q[13];
U1q(0.598704375670855*pi,1.3338756539330898*pi) q[14];
U1q(0.370977580609839*pi,1.3434606795254904*pi) q[15];
U1q(0.343772012182575*pi,1.8537432176895097*pi) q[16];
U1q(0.31603907078433*pi,0.9964328795336996*pi) q[17];
U1q(0.957052176496024*pi,1.7854483598200699*pi) q[18];
U1q(0.143599286745125*pi,1.7112174614137903*pi) q[19];
U1q(0.535531976704831*pi,0.0823877624765803*pi) q[20];
U1q(0.450840360884331*pi,1.1685979973782104*pi) q[21];
U1q(0.452289586198566*pi,0.6748348669123896*pi) q[22];
U1q(0.536163207038585*pi,0.64172079158784*pi) q[23];
U1q(0.543795004208952*pi,1.6968599147928498*pi) q[24];
U1q(0.588937844081143*pi,1.52326655933813*pi) q[25];
U1q(0.797373606144285*pi,0.31846624114096*pi) q[26];
U1q(0.421747300506736*pi,0.9344781272522003*pi) q[27];
U1q(0.552393898975304*pi,1.1999771448116299*pi) q[28];
U1q(0.751551954251477*pi,0.18026986931625988*pi) q[29];
U1q(0.29753387476027*pi,1.4686970791227*pi) q[30];
U1q(0.852931919102081*pi,1.86003624170867*pi) q[31];
U1q(0.560264051798433*pi,0.15655667061576994*pi) q[32];
U1q(0.791965782392143*pi,0.3931403491144798*pi) q[33];
U1q(0.673871081209967*pi,1.15319184918315*pi) q[34];
U1q(0.553119531730448*pi,1.2632126276030897*pi) q[35];
U1q(0.575830967543432*pi,0.9156070886775698*pi) q[36];
U1q(0.412891719037095*pi,1.4664953945421404*pi) q[37];
U1q(0.409763466763055*pi,0.0829432718874199*pi) q[38];
U1q(0.35340466981103*pi,0.0036003040249299456*pi) q[39];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[7],q[29];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[33],q[12];
RZZ(0.5*pi) q[28],q[14];
RZZ(0.5*pi) q[34],q[17];
RZZ(0.5*pi) q[31],q[18];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[39],q[21];
RZZ(0.5*pi) q[22],q[27];
RZZ(0.5*pi) q[24],q[32];
RZZ(0.5*pi) q[26],q[25];
RZZ(0.5*pi) q[30],q[38];
RZZ(0.5*pi) q[36],q[37];
U1q(0.531102554791941*pi,1.3690022672281001*pi) q[0];
U1q(0.730242506694797*pi,0.96163080887571*pi) q[1];
U1q(0.469994064604968*pi,1.18825789767462*pi) q[2];
U1q(0.370230079853431*pi,0.6577906659306301*pi) q[3];
U1q(0.278735945945891*pi,0.7864554623320998*pi) q[4];
U1q(0.587478569598522*pi,1.9312829827360307*pi) q[5];
U1q(0.811183245292989*pi,1.97640404668903*pi) q[6];
U1q(0.892146364117193*pi,1.5334575373826302*pi) q[7];
U1q(0.754383664623293*pi,0.9977737771740198*pi) q[8];
U1q(0.532558715667299*pi,1.3730882889494902*pi) q[9];
U1q(0.769704710960499*pi,1.0481047288492302*pi) q[10];
U1q(0.29753788906494*pi,0.42119995815271016*pi) q[11];
U1q(0.724226392264533*pi,0.3310839120813398*pi) q[12];
U1q(0.221048413623916*pi,0.9341853773886202*pi) q[13];
U1q(0.861086111933387*pi,1.0868362783511198*pi) q[14];
U1q(0.749278896967843*pi,0.3262150737101903*pi) q[15];
U1q(0.380662903704096*pi,0.4232492179674896*pi) q[16];
U1q(0.532688415111212*pi,1.9770659651760294*pi) q[17];
U1q(0.196472780011342*pi,0.1390495741946598*pi) q[18];
U1q(0.358239345039322*pi,0.76011468572844*pi) q[19];
U1q(0.502249509600728*pi,1.9125892573708798*pi) q[20];
U1q(0.0387417383845313*pi,0.18109012451496032*pi) q[21];
U1q(0.282010879126015*pi,0.5033328838071904*pi) q[22];
U1q(0.437740009829147*pi,0.7695714511862697*pi) q[23];
U1q(0.485084214164681*pi,1.3300298018608396*pi) q[24];
U1q(0.444288946507134*pi,0.8698780675528601*pi) q[25];
U1q(0.91992780571658*pi,0.03238349468372981*pi) q[26];
U1q(0.616854166822947*pi,1.6289838853895997*pi) q[27];
U1q(0.371604868529998*pi,1.2252206730046504*pi) q[28];
U1q(0.459082947405563*pi,1.8771762293595797*pi) q[29];
U1q(0.537767584110988*pi,0.52934325112283*pi) q[30];
U1q(0.600837518881842*pi,0.8123483143502597*pi) q[31];
U1q(0.0880005477470087*pi,0.2464764398923598*pi) q[32];
U1q(0.585402013508062*pi,0.7217511133943901*pi) q[33];
U1q(0.475235501520735*pi,1.8175919261025504*pi) q[34];
U1q(0.586814215362547*pi,1.0419135463571596*pi) q[35];
U1q(0.719205480803286*pi,1.1223669917173504*pi) q[36];
U1q(0.516710726550616*pi,0.15753548538547957*pi) q[37];
U1q(0.333632690692524*pi,0.11703905591005004*pi) q[38];
U1q(0.874496750856784*pi,1.5250572929357302*pi) q[39];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[35],q[2];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[4],q[20];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[13],q[7];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[12],q[34];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[38];
RZZ(0.5*pi) q[25],q[16];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[29],q[23];
RZZ(0.5*pi) q[30],q[28];
RZZ(0.5*pi) q[33],q[36];
U1q(0.885152901038751*pi,0.9700162103797396*pi) q[0];
U1q(0.780823072162731*pi,1.4190788053764702*pi) q[1];
U1q(0.238056366810221*pi,0.9920740749235097*pi) q[2];
U1q(0.968166762082699*pi,1.4602329413246*pi) q[3];
U1q(0.316377557349116*pi,1.2694435898749994*pi) q[4];
U1q(0.520843829606615*pi,0.04209636463182065*pi) q[5];
U1q(0.386604826743424*pi,1.0954245243166003*pi) q[6];
U1q(0.537344923213413*pi,1.7125104194321903*pi) q[7];
U1q(0.28977601266454*pi,0.6446461998606896*pi) q[8];
U1q(0.0347781327253813*pi,0.7187027908269208*pi) q[9];
U1q(0.353485945887911*pi,1.5689949207102991*pi) q[10];
U1q(0.421067879286011*pi,1.8020630416699497*pi) q[11];
U1q(0.306575214498163*pi,1.1177493726908008*pi) q[12];
U1q(0.225201480712936*pi,0.05345937032386061*pi) q[13];
U1q(0.554730860155874*pi,0.6503891273820894*pi) q[14];
U1q(0.218381053909313*pi,0.14166226172016927*pi) q[15];
U1q(0.440811961753149*pi,1.5041741033981992*pi) q[16];
U1q(0.437890642947998*pi,1.3371759468665996*pi) q[17];
U1q(0.591045856006142*pi,0.27462296940158026*pi) q[18];
U1q(0.573205122558567*pi,1.1419610010398493*pi) q[19];
U1q(0.227695101574666*pi,1.7555374814249607*pi) q[20];
U1q(0.349274174047512*pi,0.6604184904866397*pi) q[21];
U1q(0.0654363715374389*pi,0.5297394388015295*pi) q[22];
U1q(0.801011548707385*pi,1.4834967878542198*pi) q[23];
U1q(0.148221367230769*pi,0.001370522540280028*pi) q[24];
U1q(0.182052432223712*pi,0.2168023046833003*pi) q[25];
U1q(0.230876620005702*pi,1.4444720252507395*pi) q[26];
U1q(0.783445133391652*pi,1.5803133964665008*pi) q[27];
U1q(0.871348029232278*pi,0.06791636299666948*pi) q[28];
U1q(0.537087406096436*pi,0.8516179319066595*pi) q[29];
U1q(0.707713332669928*pi,1.8134249315643007*pi) q[30];
U1q(0.346238350638674*pi,0.6245003838610801*pi) q[31];
U1q(0.68723358678111*pi,0.30500032488826*pi) q[32];
U1q(0.794338446995571*pi,1.83426034698003*pi) q[33];
U1q(0.265233667333695*pi,0.4436333301633004*pi) q[34];
U1q(0.637514910666291*pi,1.2548926476567495*pi) q[35];
U1q(0.285458067964424*pi,0.028935957593779982*pi) q[36];
U1q(0.121963717460225*pi,0.5330696143305804*pi) q[37];
U1q(0.430579218307162*pi,1.8599378446109602*pi) q[38];
U1q(0.734555278483233*pi,1.8487263829301401*pi) q[39];
RZZ(0.5*pi) q[3],q[0];
RZZ(0.5*pi) q[1],q[18];
RZZ(0.5*pi) q[28],q[2];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[24],q[7];
RZZ(0.5*pi) q[32],q[8];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[36],q[10];
RZZ(0.5*pi) q[12],q[17];
RZZ(0.5*pi) q[31],q[14];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[19],q[38];
RZZ(0.5*pi) q[20],q[25];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[29],q[37];
U1q(0.665200177981121*pi,1.8852586319322509*pi) q[0];
U1q(0.376472647579849*pi,1.3663510098510994*pi) q[1];
U1q(0.315609370563584*pi,1.1807185483476008*pi) q[2];
U1q(0.451499647396782*pi,1.1597687212686996*pi) q[3];
U1q(0.391527663928452*pi,1.0031184987238007*pi) q[4];
U1q(0.637175534289751*pi,0.7487675351821004*pi) q[5];
U1q(0.774250984347045*pi,0.1389190390527002*pi) q[6];
U1q(0.257576195001025*pi,1.8464031045062992*pi) q[7];
U1q(0.264106242475745*pi,1.1028411512945002*pi) q[8];
U1q(0.46410970561912*pi,1.3693750185810991*pi) q[9];
U1q(0.198490912412824*pi,0.40415576373989914*pi) q[10];
U1q(0.112489188796263*pi,0.5324638222329892*pi) q[11];
U1q(0.492361660542303*pi,0.3098980819087007*pi) q[12];
U1q(0.733134577165919*pi,0.5925253307327996*pi) q[13];
U1q(0.634770929989139*pi,1.9958397610060992*pi) q[14];
U1q(0.520375269316947*pi,0.5950008017645008*pi) q[15];
U1q(0.616709732925855*pi,0.3558937415025998*pi) q[16];
U1q(0.255826607669828*pi,0.8985168807306998*pi) q[17];
U1q(0.141314350436382*pi,1.7914537044926497*pi) q[18];
U1q(0.555299112053535*pi,0.9774125857358005*pi) q[19];
U1q(0.362022321815337*pi,0.6415875200243999*pi) q[20];
U1q(0.643793004482177*pi,1.0324023559689*pi) q[21];
U1q(0.317409194886582*pi,1.7951018460707004*pi) q[22];
U1q(0.187082177356185*pi,0.8314999506427707*pi) q[23];
U1q(0.00945108606828878*pi,1.8622536237710996*pi) q[24];
U1q(0.525091019217719*pi,1.7660614769041008*pi) q[25];
U1q(0.532345654039026*pi,0.29673955239159966*pi) q[26];
U1q(0.670032001767865*pi,1.2516462041576997*pi) q[27];
U1q(0.318968019658916*pi,1.4572002205533003*pi) q[28];
U1q(0.38904818685647*pi,0.7603633923581992*pi) q[29];
U1q(0.500421522700364*pi,1.9255919601793998*pi) q[30];
U1q(0.74853123620209*pi,1.9884380232235994*pi) q[31];
U1q(0.428014187487473*pi,1.7646397192190602*pi) q[32];
U1q(0.120723618513644*pi,1.7176683992566595*pi) q[33];
U1q(0.707677205361066*pi,1.3285855634107993*pi) q[34];
U1q(0.696332032767999*pi,0.17253479429973062*pi) q[35];
U1q(0.941963904111163*pi,1.4174296175596002*pi) q[36];
U1q(0.651054997927093*pi,1.1590647280772703*pi) q[37];
U1q(0.805402964759632*pi,0.5692747483304306*pi) q[38];
U1q(0.660404848560161*pi,1.7817539721991498*pi) q[39];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[1],q[3];
RZZ(0.5*pi) q[22],q[2];
RZZ(0.5*pi) q[4],q[10];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[6],q[16];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[18],q[8];
RZZ(0.5*pi) q[26],q[9];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[24],q[12];
RZZ(0.5*pi) q[33],q[13];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[30],q[17];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[39],q[23];
RZZ(0.5*pi) q[25],q[38];
RZZ(0.5*pi) q[36],q[32];
RZZ(0.5*pi) q[35],q[37];
U1q(0.466661198499244*pi,0.7655478509987503*pi) q[0];
U1q(0.641277557125697*pi,0.034837182658499444*pi) q[1];
U1q(0.518912545687094*pi,0.2770222759126*pi) q[2];
U1q(0.406635415156109*pi,0.25159637239170074*pi) q[3];
U1q(0.274703494084429*pi,0.5641763669136992*pi) q[4];
U1q(0.72411783823276*pi,1.4368164767771994*pi) q[5];
U1q(0.853561001503428*pi,1.2668069734361005*pi) q[6];
U1q(0.28780542040724*pi,0.33397574306500033*pi) q[7];
U1q(0.195200010375671*pi,0.6141517665432996*pi) q[8];
U1q(0.31669311964262*pi,1.1061552270661004*pi) q[9];
U1q(0.276599827373759*pi,1.3476436602402*pi) q[10];
U1q(0.4239523223681*pi,0.2748484843372001*pi) q[11];
U1q(0.775412498388889*pi,0.16385941351430056*pi) q[12];
U1q(0.725117889908967*pi,1.5485085037375992*pi) q[13];
U1q(0.235827680530444*pi,0.48365227000249966*pi) q[14];
U1q(0.631948446070486*pi,1.2146195108488005*pi) q[15];
U1q(0.300896617746962*pi,0.30626948242369956*pi) q[16];
U1q(0.171483534367627*pi,0.45071707944689976*pi) q[17];
U1q(0.604139825670373*pi,0.014173245349200059*pi) q[18];
U1q(0.280987127105549*pi,1.9678734147925994*pi) q[19];
U1q(0.296076324353434*pi,0.7470750017689003*pi) q[20];
U1q(0.509212182407907*pi,0.2913249650931*pi) q[21];
U1q(0.412016719972715*pi,0.8284861830749009*pi) q[22];
U1q(0.816396121855198*pi,1.6941764823358003*pi) q[23];
U1q(0.458714008647674*pi,0.8789224824141009*pi) q[24];
U1q(0.707780951703639*pi,0.7895582961541994*pi) q[25];
U1q(0.357758647165178*pi,1.1634029361194003*pi) q[26];
U1q(0.34218552686649*pi,1.5986283482220998*pi) q[27];
U1q(0.648564452210976*pi,0.9566260920191993*pi) q[28];
U1q(0.895118148507188*pi,1.5846463618347002*pi) q[29];
U1q(0.604714331228608*pi,0.9398324448913993*pi) q[30];
U1q(0.973154325478944*pi,1.6724370666647008*pi) q[31];
U1q(0.75926204569379*pi,1.6296790713908393*pi) q[32];
U1q(0.104466924289589*pi,1.6647392067349998*pi) q[33];
U1q(0.0288699609920759*pi,1.6230140788669*pi) q[34];
U1q(0.603760414802403*pi,0.1953715392372004*pi) q[35];
U1q(0.728878510851044*pi,0.9810894297324992*pi) q[36];
U1q(0.82270640770628*pi,0.6521909132708501*pi) q[37];
U1q(0.653576781777895*pi,0.9024345562674991*pi) q[38];
U1q(0.176594643839318*pi,0.8728466498256005*pi) q[39];
RZZ(0.5*pi) q[24],q[0];
RZZ(0.5*pi) q[1],q[32];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[36],q[4];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[9],q[37];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[15],q[12];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[14],q[38];
RZZ(0.5*pi) q[31],q[16];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[21],q[19];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[30],q[25];
RZZ(0.5*pi) q[33],q[27];
RZZ(0.5*pi) q[34],q[28];
U1q(0.55093406744933*pi,0.9765752014154998*pi) q[0];
U1q(0.94027272665703*pi,0.020104723744900355*pi) q[1];
U1q(0.683080798205778*pi,1.8366972686994991*pi) q[2];
U1q(0.436917210776864*pi,0.9727922567376002*pi) q[3];
U1q(0.362807332585341*pi,0.3599552748286001*pi) q[4];
U1q(0.269807046223347*pi,0.48114991367659954*pi) q[5];
U1q(0.862176217332183*pi,0.14852118885499976*pi) q[6];
U1q(0.841464462306433*pi,1.7098144603956005*pi) q[7];
U1q(0.560668749827564*pi,0.6531157009465005*pi) q[8];
U1q(0.598848311203167*pi,0.996952330841701*pi) q[9];
U1q(0.673377126306918*pi,1.8128572422277003*pi) q[10];
U1q(0.203339016044197*pi,1.5330819762773*pi) q[11];
U1q(0.262001318348823*pi,1.7169293177949*pi) q[12];
U1q(0.288541576878868*pi,1.5594667773245003*pi) q[13];
U1q(0.42942112238909*pi,0.5456281406249985*pi) q[14];
U1q(0.336238559851628*pi,1.5445011656637*pi) q[15];
U1q(0.389287883309299*pi,0.2591921689075001*pi) q[16];
U1q(0.49102490712963*pi,1.655105233012499*pi) q[17];
U1q(0.33712980621334*pi,1.7742271225093997*pi) q[18];
U1q(0.161512772879083*pi,0.7742737350676983*pi) q[19];
U1q(0.506472035483343*pi,1.4599593775720017*pi) q[20];
U1q(0.285205595718648*pi,0.7312111215836001*pi) q[21];
U1q(0.687736616494732*pi,0.6061447824308992*pi) q[22];
U1q(0.294840898129407*pi,0.054903698775300214*pi) q[23];
U1q(0.35801770424758*pi,1.9037326088911009*pi) q[24];
U1q(0.565252934070297*pi,0.06500162664779907*pi) q[25];
U1q(0.553258882339331*pi,0.5563346547384995*pi) q[26];
U1q(0.122103762346313*pi,1.6316497530354006*pi) q[27];
U1q(0.359859692032276*pi,0.9406370392179007*pi) q[28];
U1q(0.285739871230825*pi,1.3609504597250996*pi) q[29];
U1q(0.296638502559999*pi,1.4805664724366991*pi) q[30];
U1q(0.40631893494169*pi,0.21810433018680087*pi) q[31];
U1q(0.241397811701256*pi,0.0045928960684005204*pi) q[32];
U1q(0.785904783073881*pi,1.1145993485963004*pi) q[33];
U1q(0.483371347501494*pi,1.0911888464986994*pi) q[34];
U1q(0.698976113965965*pi,0.08538494925569928*pi) q[35];
U1q(0.585042279271346*pi,1.8562303562269005*pi) q[36];
U1q(0.31309631673529*pi,0.7135981554008008*pi) q[37];
U1q(0.736570519703344*pi,1.3430914255798*pi) q[38];
U1q(0.190546991795309*pi,0.1641038176449996*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[38];
RZZ(0.5*pi) q[13],q[2];
RZZ(0.5*pi) q[39],q[3];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[5],q[32];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[20],q[8];
RZZ(0.5*pi) q[9],q[16];
RZZ(0.5*pi) q[10],q[21];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[36],q[17];
RZZ(0.5*pi) q[27],q[19];
RZZ(0.5*pi) q[35],q[22];
RZZ(0.5*pi) q[24],q[30];
RZZ(0.5*pi) q[34],q[25];
RZZ(0.5*pi) q[33],q[28];
RZZ(0.5*pi) q[31],q[29];
U1q(0.292787879264853*pi,0.35180399037930066*pi) q[0];
U1q(0.879076799471255*pi,0.9939362438281982*pi) q[1];
U1q(0.9587116366233*pi,0.8575745885965986*pi) q[2];
U1q(0.641875767581573*pi,0.7084824626370008*pi) q[3];
U1q(0.258239971632801*pi,0.9178699412250992*pi) q[4];
U1q(0.385599441822481*pi,1.3421974945193007*pi) q[5];
U1q(0.578403778568062*pi,1.4744263673149014*pi) q[6];
U1q(0.57200855054104*pi,0.40453723783179996*pi) q[7];
U1q(0.367529083137121*pi,1.0252043056931015*pi) q[8];
U1q(0.368779802455862*pi,1.2134923861943*pi) q[9];
U1q(0.758300187356665*pi,0.48932205775309967*pi) q[10];
U1q(0.627988864243224*pi,0.7880023143437995*pi) q[11];
U1q(0.462909980962835*pi,0.6809260299319*pi) q[12];
U1q(0.183784362919019*pi,1.2533567982546003*pi) q[13];
U1q(0.640959926097751*pi,1.1497174712960003*pi) q[14];
U1q(0.569666786546973*pi,1.3626842115857016*pi) q[15];
U1q(0.355521565261924*pi,1.2473453079378984*pi) q[16];
U1q(0.792796455290625*pi,0.22836109638160096*pi) q[17];
U1q(0.490490090086779*pi,0.7953232365911003*pi) q[18];
U1q(0.410864651851628*pi,1.000631677671901*pi) q[19];
U1q(0.311638015801257*pi,0.3699210817963987*pi) q[20];
U1q(0.683682946323619*pi,0.8046457336856996*pi) q[21];
U1q(0.413292157417189*pi,0.3622646344329006*pi) q[22];
U1q(0.738693610042349*pi,1.4711677178655993*pi) q[23];
U1q(0.358579855677908*pi,0.7246889968807011*pi) q[24];
U1q(0.584978366481756*pi,0.7500896346444001*pi) q[25];
U1q(0.410420441968389*pi,0.08046652141850075*pi) q[26];
U1q(0.100716322052129*pi,1.3423366355271007*pi) q[27];
U1q(0.401834863106804*pi,1.9679276887545*pi) q[28];
U1q(0.908044262175622*pi,0.05981562088020098*pi) q[29];
U1q(0.445580883619968*pi,1.064197508443101*pi) q[30];
U1q(0.530708583809749*pi,0.6691579742896998*pi) q[31];
U1q(0.273508200784539*pi,1.7672566777079997*pi) q[32];
U1q(0.806488620074069*pi,0.48311765534070084*pi) q[33];
U1q(0.752438934570691*pi,1.9419072835992992*pi) q[34];
U1q(0.474975747719082*pi,0.07580156793009962*pi) q[35];
U1q(0.346983335245936*pi,1.2214335859330987*pi) q[36];
U1q(0.416253232281682*pi,0.27627170981310023*pi) q[37];
U1q(0.415721617979389*pi,0.3815337993094996*pi) q[38];
U1q(0.595399298570141*pi,0.9570829321855996*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[17];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[28],q[6];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[39];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[36],q[15];
RZZ(0.5*pi) q[24],q[16];
RZZ(0.5*pi) q[26],q[18];
RZZ(0.5*pi) q[20],q[34];
RZZ(0.5*pi) q[32],q[21];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[33],q[37];
RZZ(0.5*pi) q[35],q[38];
U1q(0.842355260818823*pi,0.5422865943946995*pi) q[0];
U1q(0.430558405407104*pi,0.5425707705466998*pi) q[1];
U1q(0.755227154551869*pi,0.6808010770334008*pi) q[2];
U1q(0.57018627118262*pi,1.8354058673123994*pi) q[3];
U1q(0.441344680244376*pi,1.3335810387882994*pi) q[4];
U1q(0.489238288938196*pi,1.4737026403697016*pi) q[5];
U1q(0.548514269339652*pi,0.5605378507491992*pi) q[6];
U1q(0.598265744746993*pi,0.19048064466329961*pi) q[7];
U1q(0.193216706380552*pi,0.706008166111701*pi) q[8];
U1q(0.863690180614688*pi,1.383836130384001*pi) q[9];
U1q(0.554380834986687*pi,1.6753823039978997*pi) q[10];
U1q(0.510546673982997*pi,0.23887378189539987*pi) q[11];
U1q(0.786840316779897*pi,1.5101935836535993*pi) q[12];
U1q(0.245166480417302*pi,1.5819910507266002*pi) q[13];
U1q(0.478090812927668*pi,0.1442107459603008*pi) q[14];
U1q(0.336064906947782*pi,0.21508531088809946*pi) q[15];
U1q(0.171252492190124*pi,0.9330721029811997*pi) q[16];
U1q(0.268170284130765*pi,1.5220288287938004*pi) q[17];
U1q(0.566902759741251*pi,1.3057990075849002*pi) q[18];
U1q(0.275298573272065*pi,1.8596505034361996*pi) q[19];
U1q(0.292221141837383*pi,1.7300064315701*pi) q[20];
U1q(0.633056979641255*pi,0.2283532970620996*pi) q[21];
U1q(0.6555651990623*pi,0.3650936446107984*pi) q[22];
U1q(0.489327738030217*pi,0.5695898805279*pi) q[23];
U1q(0.868874927389783*pi,1.7037184356506998*pi) q[24];
U1q(0.535076468319017*pi,0.4635538195109987*pi) q[25];
U1q(0.352559890807553*pi,1.3294093606645987*pi) q[26];
U1q(0.33355979663554*pi,0.446830581208701*pi) q[27];
U1q(0.843752351764649*pi,1.7112705455819004*pi) q[28];
U1q(0.262207749043899*pi,1.2556131243756994*pi) q[29];
U1q(0.426082715778225*pi,0.7032662531346006*pi) q[30];
U1q(0.10275467641231*pi,1.5293138523046999*pi) q[31];
U1q(0.557265797661082*pi,0.44907981148000076*pi) q[32];
U1q(0.586934693959129*pi,0.3531873382719013*pi) q[33];
U1q(0.560028803315179*pi,1.5497354158427008*pi) q[34];
U1q(0.417507695826579*pi,1.7923135285731995*pi) q[35];
U1q(0.378411588811096*pi,0.9560172801909985*pi) q[36];
U1q(0.457108463468304*pi,0.22761216840340026*pi) q[37];
U1q(0.303132218795665*pi,0.9521890895927001*pi) q[38];
U1q(0.716087934520951*pi,1.6685485166297003*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[2],q[6];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[4],q[11];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[7],q[39];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[18],q[10];
RZZ(0.5*pi) q[12],q[27];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[21],q[14];
RZZ(0.5*pi) q[17],q[37];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[36],q[23];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[33],q[31];
U1q(0.758740731191323*pi,1.3616771493592985*pi) q[0];
U1q(0.237258946750108*pi,1.9174580676412987*pi) q[1];
U1q(0.418094024289555*pi,0.09447231837020098*pi) q[2];
U1q(0.310377637609994*pi,1.8033923623910013*pi) q[3];
U1q(0.266270753866971*pi,0.8697798649348982*pi) q[4];
U1q(0.320307925227539*pi,0.010409679946199901*pi) q[5];
U1q(0.151357234510929*pi,1.6494571317976003*pi) q[6];
U1q(0.576540820061786*pi,0.07653540212789878*pi) q[7];
U1q(0.567191821014019*pi,1.0975600227734006*pi) q[8];
U1q(0.858656960615947*pi,1.5424154042632985*pi) q[9];
U1q(0.0636424952421057*pi,1.0305909409822007*pi) q[10];
U1q(0.352096041173743*pi,1.729798118599799*pi) q[11];
U1q(0.359402187389353*pi,1.7707593974534994*pi) q[12];
U1q(0.370542297316163*pi,1.4144373026060002*pi) q[13];
U1q(0.571818770858837*pi,1.825684476001399*pi) q[14];
U1q(0.81899979754871*pi,1.6795330507358983*pi) q[15];
U1q(0.857681832251214*pi,0.4498023067582011*pi) q[16];
U1q(0.694440251780422*pi,0.6166863731837005*pi) q[17];
U1q(0.412522074888028*pi,1.9281819972526009*pi) q[18];
U1q(0.682077989452759*pi,0.4578826138849017*pi) q[19];
U1q(0.694573093464604*pi,0.49849477489449967*pi) q[20];
U1q(0.863690534494064*pi,1.3642064136690983*pi) q[21];
U1q(0.797272546483382*pi,0.019306180887699753*pi) q[22];
U1q(0.577765312807104*pi,0.01116757933539958*pi) q[23];
U1q(0.157402849048955*pi,1.6836169551842985*pi) q[24];
U1q(0.836424620414324*pi,1.8896663016106992*pi) q[25];
U1q(0.328811565817354*pi,0.5948074327823996*pi) q[26];
U1q(0.327511395250494*pi,0.0019358290314990256*pi) q[27];
U1q(0.327136650538511*pi,0.9316222621078012*pi) q[28];
U1q(0.537896337992924*pi,0.17264517935729984*pi) q[29];
U1q(0.731501671332411*pi,1.8925768192385988*pi) q[30];
U1q(0.486794956507646*pi,1.2472688011714013*pi) q[31];
U1q(0.696682759175141*pi,1.4363643358326001*pi) q[32];
U1q(0.796848985995353*pi,0.16864723953599992*pi) q[33];
U1q(0.71924494897872*pi,0.19272878547489825*pi) q[34];
U1q(0.103080277443163*pi,0.5535229262507002*pi) q[35];
U1q(0.575006876992726*pi,1.3190304498270997*pi) q[36];
U1q(0.0560763584306624*pi,1.1231367433661994*pi) q[37];
U1q(0.600011574664055*pi,0.16162470754679958*pi) q[38];
U1q(0.229480671682677*pi,1.668241138751501*pi) q[39];
rz(2.133738410606899*pi) q[0];
rz(1.4510484884619999*pi) q[1];
rz(2.108491726442299*pi) q[2];
rz(3.0298103216089984*pi) q[3];
rz(2.819179847592501*pi) q[4];
rz(0.8051098162119992*pi) q[5];
rz(0.5786742894762007*pi) q[6];
rz(2.9361514133552014*pi) q[7];
rz(1.5674261036253014*pi) q[8];
rz(3.1532582765982013*pi) q[9];
rz(3.642521680822501*pi) q[10];
rz(1.6298286309314989*pi) q[11];
rz(3.1063242513160993*pi) q[12];
rz(2.9120841172561*pi) q[13];
rz(1.5896764944844008*pi) q[14];
rz(3.2395938273701006*pi) q[15];
rz(0.6412856695800997*pi) q[16];
rz(1.5906986379625003*pi) q[17];
rz(3.8476796646157005*pi) q[18];
rz(1.5161096243719*pi) q[19];
rz(3.9865743315071995*pi) q[20];
rz(0.30644832996329896*pi) q[21];
rz(2.098325524348599*pi) q[22];
rz(0.045425693081600826*pi) q[23];
rz(0.718445232218599*pi) q[24];
rz(0.4020089172472012*pi) q[25];
rz(3.3082140652617014*pi) q[26];
rz(3.5397795922096016*pi) q[27];
rz(2.3378427824057013*pi) q[28];
rz(1.1241807299564996*pi) q[29];
rz(1.3697536566299*pi) q[30];
rz(1.6252112142818014*pi) q[31];
rz(2.982439434599801*pi) q[32];
rz(3.7098582230040016*pi) q[33];
rz(0.9011325913027015*pi) q[34];
rz(1.9247447850707005*pi) q[35];
rz(1.1527817997504002*pi) q[36];
rz(1.8559119200466014*pi) q[37];
rz(3.755240273310701*pi) q[38];
rz(0.6520748027167009*pi) q[39];
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
