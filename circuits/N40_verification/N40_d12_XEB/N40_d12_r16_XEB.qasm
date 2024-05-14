OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.620507760874761*pi,0.356100174502637*pi) q[0];
U1q(0.619799048391688*pi,0.758355353036109*pi) q[1];
U1q(0.378661555516532*pi,0.0231130699698753*pi) q[2];
U1q(0.28497007717639*pi,0.367608655342237*pi) q[3];
U1q(0.351084919643293*pi,1.4283540096420881*pi) q[4];
U1q(0.310093700599533*pi,1.9823585664923475*pi) q[5];
U1q(0.370009007809802*pi,1.721949291385139*pi) q[6];
U1q(0.406521991634917*pi,0.20730738026746*pi) q[7];
U1q(0.428171953149538*pi,1.884774647793799*pi) q[8];
U1q(0.283700911116775*pi,0.157846508031366*pi) q[9];
U1q(0.327146786335274*pi,1.280845516897306*pi) q[10];
U1q(0.773072034203088*pi,0.875308664298293*pi) q[11];
U1q(0.815126029110145*pi,1.47143216387299*pi) q[12];
U1q(0.0774885754867976*pi,0.40860645837578*pi) q[13];
U1q(0.591720692588094*pi,0.396038039228942*pi) q[14];
U1q(0.939003245355832*pi,0.388575510634206*pi) q[15];
U1q(0.485991593154961*pi,1.875225584861999*pi) q[16];
U1q(0.470415341870622*pi,1.03585618886448*pi) q[17];
U1q(0.244787679612936*pi,1.006437613975209*pi) q[18];
U1q(0.568785421147925*pi,1.06806321112929*pi) q[19];
U1q(0.546726650653724*pi,0.463645967223966*pi) q[20];
U1q(0.748560353668779*pi,0.187228802540099*pi) q[21];
U1q(0.211674263122472*pi,0.633165692350262*pi) q[22];
U1q(0.564592590059697*pi,0.31009037623812*pi) q[23];
U1q(0.417397284558802*pi,0.298619774428703*pi) q[24];
U1q(0.485148694693659*pi,1.860573824248152*pi) q[25];
U1q(0.416540486438118*pi,0.542728016634153*pi) q[26];
U1q(0.305688095006714*pi,0.241922386987648*pi) q[27];
U1q(0.829007548417348*pi,0.825279970810215*pi) q[28];
U1q(0.300074470747494*pi,1.5556561233344839*pi) q[29];
U1q(0.14444189704384*pi,0.352402027710755*pi) q[30];
U1q(0.430883693154659*pi,0.429306520342571*pi) q[31];
U1q(0.213914607651234*pi,0.9790369764672999*pi) q[32];
U1q(0.96826668519982*pi,1.117763151487466*pi) q[33];
U1q(0.858998820167058*pi,0.616856873616625*pi) q[34];
U1q(0.520694593789542*pi,0.7310217710072*pi) q[35];
U1q(0.71239668161985*pi,0.274651951468309*pi) q[36];
U1q(0.93591830417062*pi,1.16023830845249*pi) q[37];
U1q(0.470068035362701*pi,0.80755984929585*pi) q[38];
U1q(0.406961089903742*pi,0.812031530971101*pi) q[39];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[29],q[1];
RZZ(0.5*pi) q[2],q[36];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[5],q[20];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[25];
RZZ(0.5*pi) q[8],q[19];
RZZ(0.5*pi) q[35],q[9];
RZZ(0.5*pi) q[39],q[10];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[15],q[14];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[21],q[22];
RZZ(0.5*pi) q[38],q[23];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[37],q[26];
RZZ(0.5*pi) q[30],q[27];
RZZ(0.5*pi) q[33],q[34];
U1q(0.462021948437363*pi,1.9817529653057702*pi) q[0];
U1q(0.485226432721951*pi,1.592637770667016*pi) q[1];
U1q(0.854338076336646*pi,1.20340803550034*pi) q[2];
U1q(0.252893969080612*pi,0.6092670386174599*pi) q[3];
U1q(0.182078735918641*pi,1.3763316296007*pi) q[4];
U1q(0.225987591938826*pi,1.38141043773085*pi) q[5];
U1q(0.447123161354731*pi,1.23867887916265*pi) q[6];
U1q(0.571872013821787*pi,1.97510434064815*pi) q[7];
U1q(0.929190895164449*pi,1.33372108455631*pi) q[8];
U1q(0.199746831098509*pi,1.5127447464213999*pi) q[9];
U1q(0.52081071722185*pi,0.4292661543237699*pi) q[10];
U1q(0.412328017452744*pi,0.7031916280716599*pi) q[11];
U1q(0.235771443879075*pi,1.9362035203117682*pi) q[12];
U1q(0.896226892020864*pi,0.17418538607285994*pi) q[13];
U1q(0.204984456320895*pi,0.4677934345805801*pi) q[14];
U1q(0.248624088437592*pi,1.56846042904961*pi) q[15];
U1q(0.407869355636081*pi,1.238730684722554*pi) q[16];
U1q(0.698448611993657*pi,0.43373874652533995*pi) q[17];
U1q(0.202646310873049*pi,0.69179281897833*pi) q[18];
U1q(0.700140247627448*pi,0.204855049382754*pi) q[19];
U1q(0.449604493134574*pi,0.7002297332093399*pi) q[20];
U1q(0.214902855450982*pi,0.2822771702304301*pi) q[21];
U1q(0.495819716046228*pi,0.68601409705107*pi) q[22];
U1q(0.500057257913035*pi,1.79562283713557*pi) q[23];
U1q(0.620648829931732*pi,0.018284734774159928*pi) q[24];
U1q(0.313771297217902*pi,1.61675316756532*pi) q[25];
U1q(0.419517871201488*pi,0.010188826357929948*pi) q[26];
U1q(0.674478336707779*pi,1.079226640990239*pi) q[27];
U1q(0.279735265170186*pi,1.362631090299832*pi) q[28];
U1q(0.403867367792537*pi,1.010507790647972*pi) q[29];
U1q(0.333161995427337*pi,1.01721025513914*pi) q[30];
U1q(0.49844423051405*pi,1.69959520554917*pi) q[31];
U1q(0.546387638328036*pi,0.8809264115236402*pi) q[32];
U1q(0.460721314444644*pi,0.8224107065417798*pi) q[33];
U1q(0.25408125049714*pi,1.370888436525472*pi) q[34];
U1q(0.388968310655034*pi,1.7675792620702104*pi) q[35];
U1q(0.543882378459575*pi,0.7598085093431299*pi) q[36];
U1q(0.887819019973517*pi,0.540852820618243*pi) q[37];
U1q(0.592435376045326*pi,1.41377149614907*pi) q[38];
U1q(0.341792833023236*pi,0.4429234978094301*pi) q[39];
RZZ(0.5*pi) q[5],q[0];
RZZ(0.5*pi) q[25],q[1];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[3],q[9];
RZZ(0.5*pi) q[28],q[4];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[7],q[39];
RZZ(0.5*pi) q[10],q[12];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[14],q[38];
RZZ(0.5*pi) q[15],q[26];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[18],q[22];
RZZ(0.5*pi) q[37],q[19];
RZZ(0.5*pi) q[24],q[20];
RZZ(0.5*pi) q[21],q[29];
RZZ(0.5*pi) q[31],q[30];
RZZ(0.5*pi) q[32],q[34];
RZZ(0.5*pi) q[35],q[36];
U1q(0.411550357212292*pi,0.7092941940264499*pi) q[0];
U1q(0.529250295110207*pi,0.18297434132824986*pi) q[1];
U1q(0.341679593931523*pi,0.17415610789349012*pi) q[2];
U1q(0.78211913871263*pi,1.5305448776927904*pi) q[3];
U1q(0.0609409259956717*pi,1.3599621271819204*pi) q[4];
U1q(0.505659372309422*pi,0.07359414423883992*pi) q[5];
U1q(0.0978590853655209*pi,1.1960126379209601*pi) q[6];
U1q(0.406703540277042*pi,1.9044278072027199*pi) q[7];
U1q(0.415426118933378*pi,0.1800748294425798*pi) q[8];
U1q(0.25789757828886*pi,0.42486921371248965*pi) q[9];
U1q(0.193923286897965*pi,1.4071166428109798*pi) q[10];
U1q(0.157340609067275*pi,1.0744022149000099*pi) q[11];
U1q(0.810974571163955*pi,1.87182432575259*pi) q[12];
U1q(0.605973156320066*pi,0.1085711433537*pi) q[13];
U1q(0.312441170535438*pi,1.5849788842257402*pi) q[14];
U1q(0.376909033508843*pi,1.6412518143864698*pi) q[15];
U1q(0.261647003044213*pi,1.42345366182591*pi) q[16];
U1q(0.800733288990866*pi,0.17492405084961993*pi) q[17];
U1q(0.0947889045076442*pi,1.08356993468011*pi) q[18];
U1q(0.558032564948029*pi,1.63366272387487*pi) q[19];
U1q(0.26808265696353*pi,1.4608195552951901*pi) q[20];
U1q(0.327599875770836*pi,0.9078620760435299*pi) q[21];
U1q(0.39695727266366*pi,0.6328804816553104*pi) q[22];
U1q(0.715120122644801*pi,0.8566328621383397*pi) q[23];
U1q(0.498857962482251*pi,1.3335872681346403*pi) q[24];
U1q(0.684526970809389*pi,1.2710374356871696*pi) q[25];
U1q(0.329014145408642*pi,1.2288493240898104*pi) q[26];
U1q(0.689550307369828*pi,0.08467276911240007*pi) q[27];
U1q(0.247364287906012*pi,0.58606671403414*pi) q[28];
U1q(0.59274527710433*pi,0.7769117439156301*pi) q[29];
U1q(0.830484910736031*pi,1.8681475745315597*pi) q[30];
U1q(0.872128522847197*pi,0.5084409173505602*pi) q[31];
U1q(0.405147295970599*pi,1.6222760916425303*pi) q[32];
U1q(0.583797140374781*pi,0.21372323268423976*pi) q[33];
U1q(0.855946385222866*pi,1.26204712401332*pi) q[34];
U1q(0.748436727348069*pi,0.8263382523489904*pi) q[35];
U1q(0.552347727929581*pi,1.1860884600106596*pi) q[36];
U1q(0.346929425858001*pi,0.72664819192708*pi) q[37];
U1q(0.757971402890051*pi,1.75097590423088*pi) q[38];
U1q(0.566187741923159*pi,1.6784578418468197*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[32],q[4];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[21],q[12];
RZZ(0.5*pi) q[13],q[24];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[16],q[35];
RZZ(0.5*pi) q[18],q[38];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[25],q[30];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[33],q[28];
RZZ(0.5*pi) q[37],q[29];
RZZ(0.5*pi) q[39],q[31];
U1q(0.525996788444143*pi,0.2946791560873896*pi) q[0];
U1q(0.953643451097722*pi,1.2774709212915099*pi) q[1];
U1q(0.577161570042793*pi,1.92037904139202*pi) q[2];
U1q(0.294790878306451*pi,1.7314510404394499*pi) q[3];
U1q(0.466975586001318*pi,0.6388357000699996*pi) q[4];
U1q(0.370448868796013*pi,0.6936558873913796*pi) q[5];
U1q(0.528107597926089*pi,0.6506113796116404*pi) q[6];
U1q(0.566608997227158*pi,0.15228357285046012*pi) q[7];
U1q(0.556244631305034*pi,0.13752238453958032*pi) q[8];
U1q(0.646224787036759*pi,1.4154973785346803*pi) q[9];
U1q(0.195780009253643*pi,0.26745413894335*pi) q[10];
U1q(0.571322828659424*pi,1.92833228362574*pi) q[11];
U1q(0.524556305398886*pi,1.48172045030213*pi) q[12];
U1q(0.291953350605325*pi,1.25946594686667*pi) q[13];
U1q(0.500749599286972*pi,1.22688870780462*pi) q[14];
U1q(0.474414052845625*pi,0.41535144440824023*pi) q[15];
U1q(0.31163562185874*pi,1.8876155911114898*pi) q[16];
U1q(0.674185876609108*pi,1.9468057062378001*pi) q[17];
U1q(0.841368280418958*pi,0.22553888562990032*pi) q[18];
U1q(0.789696094503224*pi,1.1961058972225196*pi) q[19];
U1q(0.463986616673614*pi,1.1029776097840802*pi) q[20];
U1q(0.805570368360083*pi,1.8576855587617*pi) q[21];
U1q(0.554422941408599*pi,1.5103223010284*pi) q[22];
U1q(0.704376241256369*pi,0.21073756572850044*pi) q[23];
U1q(0.760022101961222*pi,0.3372045776261796*pi) q[24];
U1q(0.440503877121264*pi,1.68433859110511*pi) q[25];
U1q(0.808899547537406*pi,0.01417682832576972*pi) q[26];
U1q(0.395607954284636*pi,0.9336783861109303*pi) q[27];
U1q(0.342005046244354*pi,1.3705039202566702*pi) q[28];
U1q(0.63795662235558*pi,0.5929636601300103*pi) q[29];
U1q(0.888303237873017*pi,1.8163626953111294*pi) q[30];
U1q(0.489548610972216*pi,1.48707437667714*pi) q[31];
U1q(0.769779328522122*pi,0.34755854542417985*pi) q[32];
U1q(0.476040699935059*pi,0.3575242016732796*pi) q[33];
U1q(0.67610792194406*pi,1.8696056477331204*pi) q[34];
U1q(0.407136122234968*pi,1.6436439273909897*pi) q[35];
U1q(0.552578666259752*pi,0.36316326815648026*pi) q[36];
U1q(0.511235781179927*pi,0.9683848913746398*pi) q[37];
U1q(0.824733132504413*pi,1.2259964326876602*pi) q[38];
U1q(0.376587289872902*pi,1.7308050702215398*pi) q[39];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[1],q[23];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[3],q[20];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[28];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[9],q[31];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[18],q[19];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[25],q[27];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[32],q[38];
RZZ(0.5*pi) q[35],q[34];
U1q(0.365569482342819*pi,0.8905834829005297*pi) q[0];
U1q(0.636602286057619*pi,0.3753410901741203*pi) q[1];
U1q(0.0991241369454074*pi,0.17843594136190966*pi) q[2];
U1q(0.56335046484303*pi,0.39536785773426963*pi) q[3];
U1q(0.68353287490545*pi,0.11383972504193007*pi) q[4];
U1q(0.380259597876318*pi,0.4592735894711808*pi) q[5];
U1q(0.396156600996363*pi,0.2609929565724691*pi) q[6];
U1q(0.658430530430774*pi,0.1481078726072198*pi) q[7];
U1q(0.663269392138606*pi,0.3555561014515405*pi) q[8];
U1q(0.491668979096166*pi,0.26766228147169*pi) q[9];
U1q(0.389909467463371*pi,1.7670956783268092*pi) q[10];
U1q(0.417368736401518*pi,1.9247055269779292*pi) q[11];
U1q(0.615097472718832*pi,1.8243101186660002*pi) q[12];
U1q(0.409459457080433*pi,1.2942784634852806*pi) q[13];
U1q(0.445640276074383*pi,0.8196170916710095*pi) q[14];
U1q(0.12724764153235*pi,1.4054593237445*pi) q[15];
U1q(0.436878976003078*pi,0.018201772292769647*pi) q[16];
U1q(0.787631783508936*pi,0.10959832698063021*pi) q[17];
U1q(0.515702511827563*pi,0.9940280017614*pi) q[18];
U1q(0.308286540160779*pi,0.3541199051769297*pi) q[19];
U1q(0.243692170333621*pi,1.7066621574699994*pi) q[20];
U1q(0.33847292814114*pi,0.6647967420504903*pi) q[21];
U1q(0.400596320075959*pi,0.39410295796201034*pi) q[22];
U1q(0.400495578726231*pi,0.30174487533640004*pi) q[23];
U1q(0.842398142983119*pi,0.29460937618095073*pi) q[24];
U1q(0.316249790045297*pi,0.4344062486959608*pi) q[25];
U1q(0.356352505075831*pi,0.4223708242897102*pi) q[26];
U1q(0.379612747759374*pi,1.40450424969979*pi) q[27];
U1q(0.75450008372504*pi,1.7435088063146402*pi) q[28];
U1q(0.363197579665466*pi,0.08699735788746032*pi) q[29];
U1q(0.63466586309476*pi,1.9493004127816995*pi) q[30];
U1q(0.761542610839634*pi,0.6497190259533898*pi) q[31];
U1q(0.648977159537367*pi,0.7072781604498495*pi) q[32];
U1q(0.66050454729361*pi,1.6221675231071*pi) q[33];
U1q(0.931032289451926*pi,0.25836743468957035*pi) q[34];
U1q(0.870824394514863*pi,0.7880311364161603*pi) q[35];
U1q(0.270814817552136*pi,0.10950132358980014*pi) q[36];
U1q(0.532288555684434*pi,1.5144440587085697*pi) q[37];
U1q(0.192446534285776*pi,0.021723938103459517*pi) q[38];
U1q(0.561305198791325*pi,0.6169295490539195*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[30];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[37],q[8];
RZZ(0.5*pi) q[35],q[10];
RZZ(0.5*pi) q[11],q[34];
RZZ(0.5*pi) q[19],q[12];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[21],q[23];
RZZ(0.5*pi) q[25],q[31];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[32],q[29];
RZZ(0.5*pi) q[33],q[39];
RZZ(0.5*pi) q[38],q[36];
U1q(0.30724958718191*pi,0.5190187663807908*pi) q[0];
U1q(0.223448843322869*pi,0.7609517773522008*pi) q[1];
U1q(0.147037687280199*pi,0.9142354014798002*pi) q[2];
U1q(0.841584867401343*pi,0.7110291100115997*pi) q[3];
U1q(0.498137368167144*pi,1.7925787130995996*pi) q[4];
U1q(0.545628954220577*pi,0.14425091171439952*pi) q[5];
U1q(0.901742515225554*pi,1.2662923286225993*pi) q[6];
U1q(0.208154911871176*pi,0.8919599579869999*pi) q[7];
U1q(0.654313448621435*pi,1.7707207725773007*pi) q[8];
U1q(0.904887416973439*pi,0.8781994636966903*pi) q[9];
U1q(0.692337058674725*pi,1.0093835070819992*pi) q[10];
U1q(0.156766413284664*pi,1.8932402720569002*pi) q[11];
U1q(0.281370146381566*pi,0.7121678259741904*pi) q[12];
U1q(0.436837040924913*pi,1.0581492788089992*pi) q[13];
U1q(0.529339621754661*pi,0.8819816866420993*pi) q[14];
U1q(0.370472515730167*pi,0.45772751802610046*pi) q[15];
U1q(0.403366145414822*pi,0.23027997313350035*pi) q[16];
U1q(0.434681435548681*pi,0.05781072305870971*pi) q[17];
U1q(0.473792001905669*pi,1.0112717280980998*pi) q[18];
U1q(0.699923701644702*pi,1.2593306630770993*pi) q[19];
U1q(0.562775295156683*pi,0.9461445658384999*pi) q[20];
U1q(0.551918748882935*pi,1.5605293464297603*pi) q[21];
U1q(0.584642083867122*pi,1.6148701756787993*pi) q[22];
U1q(0.204551203980534*pi,1.8781150750421993*pi) q[23];
U1q(0.750849138243871*pi,1.2073825421545994*pi) q[24];
U1q(0.695081562671721*pi,1.5855319894101996*pi) q[25];
U1q(0.279613830868818*pi,1.5184583120583994*pi) q[26];
U1q(0.561358760838124*pi,1.2331766458580393*pi) q[27];
U1q(0.563394942236308*pi,1.2068791779074992*pi) q[28];
U1q(0.536941971169471*pi,1.8273308270073993*pi) q[29];
U1q(0.200634116932946*pi,1.8567278994079004*pi) q[30];
U1q(0.858090774464936*pi,0.05576715293717971*pi) q[31];
U1q(0.266768768712309*pi,0.7654760306299995*pi) q[32];
U1q(0.956621765713215*pi,1.3065368240576998*pi) q[33];
U1q(0.41721472191219*pi,1.7180509761289997*pi) q[34];
U1q(0.520014709851414*pi,1.1469858714472991*pi) q[35];
U1q(0.321254449915585*pi,0.29911278617849923*pi) q[36];
U1q(0.689972852701747*pi,0.7076286955809197*pi) q[37];
U1q(0.653297654539935*pi,0.6177412854202*pi) q[38];
U1q(0.251667890476192*pi,1.7247357196043005*pi) q[39];
RZZ(0.5*pi) q[0],q[24];
RZZ(0.5*pi) q[30],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[6],q[12];
RZZ(0.5*pi) q[7],q[23];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[37],q[10];
RZZ(0.5*pi) q[11],q[19];
RZZ(0.5*pi) q[22],q[14];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[25],q[16];
RZZ(0.5*pi) q[17],q[29];
RZZ(0.5*pi) q[18],q[34];
RZZ(0.5*pi) q[21],q[20];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[32],q[36];
RZZ(0.5*pi) q[39],q[35];
U1q(0.876470900562262*pi,0.9765850251532004*pi) q[0];
U1q(0.305234034784696*pi,0.19468749720279988*pi) q[1];
U1q(0.45876067796967*pi,0.7337087222848009*pi) q[2];
U1q(0.609528333995408*pi,1.4505499910422*pi) q[3];
U1q(0.474094740686607*pi,0.6681677467390994*pi) q[4];
U1q(0.191741402092622*pi,0.4021911261491997*pi) q[5];
U1q(0.731596219415588*pi,0.6807435995820992*pi) q[6];
U1q(0.233711951038901*pi,1.5338518811635993*pi) q[7];
U1q(0.286294464925619*pi,0.5102935209070996*pi) q[8];
U1q(0.502013690947633*pi,0.29680295642229915*pi) q[9];
U1q(0.49821912870247*pi,1.0329356023794993*pi) q[10];
U1q(0.456474582220301*pi,1.5151970291750008*pi) q[11];
U1q(0.189458883425762*pi,0.4586730218463604*pi) q[12];
U1q(0.815214352412018*pi,1.0691890793941*pi) q[13];
U1q(0.295053245128633*pi,0.6832231325234996*pi) q[14];
U1q(0.741192131103972*pi,1.2175906146506001*pi) q[15];
U1q(0.651686702495887*pi,1.6382465079075992*pi) q[16];
U1q(0.526163358773474*pi,1.4195140163523998*pi) q[17];
U1q(0.401615550552043*pi,1.5784513672637992*pi) q[18];
U1q(0.596562781738989*pi,0.01784306356952925*pi) q[19];
U1q(0.470422996335758*pi,1.2767760721543002*pi) q[20];
U1q(0.563793726668006*pi,1.1243545928863998*pi) q[21];
U1q(0.420237181525779*pi,0.39636965718909956*pi) q[22];
U1q(0.511344355197821*pi,0.2739984647122*pi) q[23];
U1q(0.904842599249931*pi,0.06530618341549932*pi) q[24];
U1q(0.447360326286863*pi,0.4827448690981999*pi) q[25];
U1q(0.384387994538098*pi,1.1994098141005*pi) q[26];
U1q(0.740651834843377*pi,1.5369898301347007*pi) q[27];
U1q(0.646576129102342*pi,0.33703506694820007*pi) q[28];
U1q(0.599982879917889*pi,0.21388607617769928*pi) q[29];
U1q(0.579187436450993*pi,0.2699824986103998*pi) q[30];
U1q(0.401499158537794*pi,0.27585157615203926*pi) q[31];
U1q(0.648376842009964*pi,1.4794047481956998*pi) q[32];
U1q(0.606905240415758*pi,1.1728445355504*pi) q[33];
U1q(0.14693608377027*pi,1.5471190878377001*pi) q[34];
U1q(0.410698227901807*pi,0.3294433675825008*pi) q[35];
U1q(0.0797450053759819*pi,1.8428838282601987*pi) q[36];
U1q(0.405844746442979*pi,1.0154116835212008*pi) q[37];
U1q(0.48950776965873*pi,1.9733532306931991*pi) q[38];
U1q(0.311229261147258*pi,1.2800319113468*pi) q[39];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[37],q[5];
RZZ(0.5*pi) q[6],q[34];
RZZ(0.5*pi) q[7],q[30];
RZZ(0.5*pi) q[16],q[9];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[28],q[12];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[15],q[24];
RZZ(0.5*pi) q[17],q[22];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[31],q[19];
RZZ(0.5*pi) q[21],q[26];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[32],q[39];
U1q(0.624812748266327*pi,0.9954918496861005*pi) q[0];
U1q(0.404791519366792*pi,1.0768510063844001*pi) q[1];
U1q(0.337271374881144*pi,0.5010099387656002*pi) q[2];
U1q(0.653590704806983*pi,0.9402821051996995*pi) q[3];
U1q(0.535792221737046*pi,0.5509638987622001*pi) q[4];
U1q(0.677806823404241*pi,1.5860252222806999*pi) q[5];
U1q(0.224844166015011*pi,1.6694574196259993*pi) q[6];
U1q(0.706660461743147*pi,1.7952259801243002*pi) q[7];
U1q(0.496844875991099*pi,1.5167061684452001*pi) q[8];
U1q(0.552431796056042*pi,1.1064028077638*pi) q[9];
U1q(0.271174839829564*pi,1.8584845791008995*pi) q[10];
U1q(0.340904706503326*pi,0.2241365284924992*pi) q[11];
U1q(0.776200417238166*pi,0.9071990877085998*pi) q[12];
U1q(0.5637808163841*pi,1.5185316773733*pi) q[13];
U1q(0.700909559923321*pi,1.6950794684439998*pi) q[14];
U1q(0.665608560483285*pi,0.44277549092630153*pi) q[15];
U1q(0.732774232012382*pi,0.2885926381719006*pi) q[16];
U1q(0.640208464023907*pi,0.3897923481092995*pi) q[17];
U1q(0.19726404169936*pi,1.7545454556881985*pi) q[18];
U1q(0.875181085975353*pi,1.6242359512272007*pi) q[19];
U1q(0.481794809559608*pi,0.44759626341840075*pi) q[20];
U1q(0.642641409961867*pi,0.1726783783462995*pi) q[21];
U1q(0.596696199109501*pi,0.7172874695834004*pi) q[22];
U1q(0.531339732964745*pi,1.8338755565482998*pi) q[23];
U1q(0.654081017242068*pi,1.0500732003681996*pi) q[24];
U1q(0.571157862007179*pi,1.3203763981849992*pi) q[25];
U1q(0.286921722857564*pi,1.4030259085214993*pi) q[26];
U1q(0.466667754984683*pi,0.020983512628900414*pi) q[27];
U1q(0.321636373797702*pi,0.6981203885205005*pi) q[28];
U1q(0.0664956432962207*pi,0.6656961254294007*pi) q[29];
U1q(0.620672194282289*pi,0.41771406127010025*pi) q[30];
U1q(0.167000406549049*pi,0.7549028967862004*pi) q[31];
U1q(0.514360492823624*pi,0.9014954372407011*pi) q[32];
U1q(0.630642536663309*pi,1.2312649203358*pi) q[33];
U1q(0.887398198621142*pi,1.6529168744072003*pi) q[34];
U1q(0.409360579017397*pi,0.7370927018503011*pi) q[35];
U1q(0.330548389555933*pi,1.4936960177448988*pi) q[36];
U1q(0.516022329952527*pi,0.2944442114678001*pi) q[37];
U1q(0.587026089523937*pi,1.3352245370342999*pi) q[38];
U1q(0.330302154784692*pi,0.5512618833699001*pi) q[39];
RZZ(0.5*pi) q[0],q[12];
RZZ(0.5*pi) q[17],q[1];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[6],q[3];
RZZ(0.5*pi) q[11],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[10],q[38];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[14],q[34];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[37],q[16];
RZZ(0.5*pi) q[32],q[18];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[25],q[20];
RZZ(0.5*pi) q[21],q[28];
RZZ(0.5*pi) q[22],q[27];
RZZ(0.5*pi) q[31],q[23];
U1q(0.887908498395724*pi,0.34886349380979986*pi) q[0];
U1q(0.428546528382974*pi,0.029796418496200516*pi) q[1];
U1q(0.536028328133482*pi,0.7399689809030008*pi) q[2];
U1q(0.588380971076675*pi,1.1540902203772987*pi) q[3];
U1q(0.6923971488116*pi,0.1321982099987995*pi) q[4];
U1q(0.403234095097688*pi,0.9198549966905993*pi) q[5];
U1q(0.371730926636397*pi,0.6709888621069986*pi) q[6];
U1q(0.705087435488395*pi,1.411690004544301*pi) q[7];
U1q(0.608363469341936*pi,0.1773764586606994*pi) q[8];
U1q(0.425126908151857*pi,1.7965678412793995*pi) q[9];
U1q(0.450485240949986*pi,1.8108751716281013*pi) q[10];
U1q(0.493521178040013*pi,0.19263394016070023*pi) q[11];
U1q(0.55759434269192*pi,1.5215034982819997*pi) q[12];
U1q(0.326434886326937*pi,0.019826588382999333*pi) q[13];
U1q(0.358057552789883*pi,1.6063994347937012*pi) q[14];
U1q(0.724965703734668*pi,0.2741163621677991*pi) q[15];
U1q(0.609243254804503*pi,0.5889595064134987*pi) q[16];
U1q(0.824378252526129*pi,1.7336101678318983*pi) q[17];
U1q(0.752861049061856*pi,0.1881038525357006*pi) q[18];
U1q(0.653392623002245*pi,1.2247632200851992*pi) q[19];
U1q(0.709159126532226*pi,1.4349122798090015*pi) q[20];
U1q(0.88541621491501*pi,1.4922647310753003*pi) q[21];
U1q(0.110895237548924*pi,0.13651029046119945*pi) q[22];
U1q(0.605072030049449*pi,1.4333212101224007*pi) q[23];
U1q(0.0731286607619635*pi,1.5317571743365015*pi) q[24];
U1q(0.328328741449419*pi,1.5408605558005988*pi) q[25];
U1q(0.0931005918798962*pi,1.7292675261812*pi) q[26];
U1q(0.658907188566536*pi,1.9219071809497983*pi) q[27];
U1q(0.485156916005085*pi,0.37963550503830135*pi) q[28];
U1q(0.64272438458584*pi,0.7452316773909011*pi) q[29];
U1q(0.417406446996446*pi,1.7056602692575993*pi) q[30];
U1q(0.828612006735533*pi,0.33700260069070076*pi) q[31];
U1q(0.464305066852631*pi,0.09520910878750044*pi) q[32];
U1q(0.691782653231079*pi,0.6840385232737987*pi) q[33];
U1q(0.25532665933294*pi,1.504174723908399*pi) q[34];
U1q(0.696382423934358*pi,1.744566814337901*pi) q[35];
U1q(0.390303197929708*pi,0.8441356252350012*pi) q[36];
U1q(0.878885883694829*pi,1.1872313637532983*pi) q[37];
U1q(0.937916812474135*pi,1.5263285744119006*pi) q[38];
U1q(0.388294634347473*pi,0.6961860603848997*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[3],q[27];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[6],q[29];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[16],q[10];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[14],q[23];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[25],q[26];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[38],q[30];
U1q(0.28305767656566*pi,0.16523565114599847*pi) q[0];
U1q(0.211274892644552*pi,0.9303702243474987*pi) q[1];
U1q(0.152286146172466*pi,1.5509211229574014*pi) q[2];
U1q(0.331842106460215*pi,1.0388612620377984*pi) q[3];
U1q(0.62956190233151*pi,1.9131420610378989*pi) q[4];
U1q(0.540489399864133*pi,1.4362138970355005*pi) q[5];
U1q(0.438152506781051*pi,0.7114050808574*pi) q[6];
U1q(0.497453169815787*pi,1.108815794563899*pi) q[7];
U1q(0.482068713227885*pi,1.2658183023847016*pi) q[8];
U1q(0.567880899212021*pi,0.1071470068339*pi) q[9];
U1q(0.260062568126189*pi,0.5423480341702991*pi) q[10];
U1q(0.478558548462639*pi,0.7552938684301012*pi) q[11];
U1q(0.268618387956605*pi,0.6989471034319994*pi) q[12];
U1q(0.689151384749913*pi,1.5812818982951988*pi) q[13];
U1q(0.767934045062861*pi,0.9418853753538983*pi) q[14];
U1q(0.403081579868446*pi,1.849134762172099*pi) q[15];
U1q(0.0832910305327961*pi,1.1123828947061014*pi) q[16];
U1q(0.792059721797633*pi,1.5920405593039*pi) q[17];
U1q(0.58824526334193*pi,0.28968214186789965*pi) q[18];
U1q(0.638535222879697*pi,1.6315508028652985*pi) q[19];
U1q(0.184872060308787*pi,0.20959892717910122*pi) q[20];
U1q(0.860853666407623*pi,1.1313825327972005*pi) q[21];
U1q(0.738803270949838*pi,1.8484961136710005*pi) q[22];
U1q(0.725446373459934*pi,0.8345626663668*pi) q[23];
U1q(0.482402649541437*pi,1.5165814213100006*pi) q[24];
U1q(0.6019144155892*pi,1.171829522877001*pi) q[25];
U1q(0.780865526698482*pi,1.1677681924916001*pi) q[26];
U1q(0.500247395096667*pi,0.6856632702425003*pi) q[27];
U1q(0.331889790619353*pi,1.3480683468685015*pi) q[28];
U1q(0.700791475504137*pi,0.023028832033698876*pi) q[29];
U1q(0.781713811420763*pi,0.6720326917568009*pi) q[30];
U1q(0.385679934684624*pi,1.993014663879901*pi) q[31];
U1q(0.160217383472115*pi,1.2877359018261991*pi) q[32];
U1q(0.965302108930386*pi,1.1703351035266998*pi) q[33];
U1q(0.68501354680621*pi,1.1119940176502006*pi) q[34];
U1q(0.544504765099902*pi,1.4026626515782006*pi) q[35];
U1q(0.788785249003236*pi,0.5738935734437014*pi) q[36];
U1q(0.245810275790166*pi,0.32590032009689907*pi) q[37];
U1q(0.235685860252628*pi,0.6142186475884017*pi) q[38];
U1q(0.364452735159617*pi,1.5515399904869014*pi) q[39];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[38],q[1];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[8],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[9],q[36];
RZZ(0.5*pi) q[33],q[10];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[13],q[22];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[24],q[27];
RZZ(0.5*pi) q[32],q[25];
RZZ(0.5*pi) q[29],q[35];
RZZ(0.5*pi) q[39],q[34];
U1q(0.275381471474313*pi,1.2102235975333997*pi) q[0];
U1q(0.643233548419656*pi,1.8087653799628*pi) q[1];
U1q(0.213521142647021*pi,0.17051756186420164*pi) q[2];
U1q(0.301687516030537*pi,1.6655847555665986*pi) q[3];
U1q(0.453012005937764*pi,0.6918431110324015*pi) q[4];
U1q(0.551996284891922*pi,0.155388428957*pi) q[5];
U1q(0.676933929953023*pi,1.5172033631787016*pi) q[6];
U1q(0.291930735540094*pi,0.10197706764870063*pi) q[7];
U1q(0.225133258770975*pi,1.5499584683407015*pi) q[8];
U1q(0.833565461667438*pi,0.8319962312095992*pi) q[9];
U1q(0.221870669910264*pi,1.6659252045006987*pi) q[10];
U1q(0.390868982190087*pi,1.1129690670574988*pi) q[11];
U1q(0.250163352565454*pi,1.1000838105575994*pi) q[12];
U1q(0.25453453125334*pi,0.2515030593059002*pi) q[13];
U1q(0.683031641595865*pi,0.46038427065570176*pi) q[14];
U1q(0.762831378629513*pi,1.3922887361349012*pi) q[15];
U1q(0.254162581943467*pi,0.7025802926894009*pi) q[16];
U1q(0.614251016674895*pi,1.3370693953899995*pi) q[17];
U1q(0.518113734769046*pi,0.06077163532500052*pi) q[18];
U1q(0.0966835779848781*pi,1.2605141290825017*pi) q[19];
U1q(0.159374745088762*pi,1.2313514931983995*pi) q[20];
U1q(0.773863621254982*pi,0.12175209303309842*pi) q[21];
U1q(0.887978409352773*pi,0.012762611590499517*pi) q[22];
U1q(0.377976437811129*pi,0.5872531055762984*pi) q[23];
U1q(0.520368960707361*pi,0.1674638362186016*pi) q[24];
U1q(0.372629351080029*pi,1.6721976165963*pi) q[25];
U1q(0.149706926568739*pi,0.026110103324700873*pi) q[26];
U1q(0.310352472226264*pi,0.1371658973853016*pi) q[27];
U1q(0.869099594462037*pi,1.8295629474597987*pi) q[28];
U1q(0.443696124493795*pi,1.7927531761554008*pi) q[29];
U1q(0.644511813979707*pi,0.7801635501371003*pi) q[30];
U1q(0.75820741254255*pi,1.9545376314971996*pi) q[31];
U1q(0.34835007491249*pi,1.5401584329097986*pi) q[32];
U1q(0.739857094270555*pi,1.3123714205725996*pi) q[33];
U1q(0.632940205018611*pi,1.2777076943539*pi) q[34];
U1q(0.292578445595999*pi,1.3106848242726983*pi) q[35];
U1q(0.648881262809235*pi,0.6985852590849007*pi) q[36];
U1q(0.206186179820753*pi,0.23761047529769996*pi) q[37];
U1q(0.43373444670191*pi,1.7956025972116016*pi) q[38];
U1q(0.772154354579037*pi,0.7895154082182003*pi) q[39];
RZZ(0.5*pi) q[0],q[30];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[26],q[3];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[5],q[27];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[39],q[9];
RZZ(0.5*pi) q[10],q[22];
RZZ(0.5*pi) q[14],q[12];
RZZ(0.5*pi) q[17],q[13];
RZZ(0.5*pi) q[32],q[16];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[21],q[25];
RZZ(0.5*pi) q[37],q[23];
RZZ(0.5*pi) q[38],q[24];
RZZ(0.5*pi) q[36],q[34];
U1q(0.670808549008863*pi,0.5502395314316999*pi) q[0];
U1q(0.344447216476877*pi,1.9694943410801997*pi) q[1];
U1q(0.75501280009414*pi,0.770283371058099*pi) q[2];
U1q(0.517617667070021*pi,1.7542717532305012*pi) q[3];
U1q(0.638987256564767*pi,0.4293196167763007*pi) q[4];
U1q(0.662014925385168*pi,0.18046026279180083*pi) q[5];
U1q(0.716904542200855*pi,0.19475463431729878*pi) q[6];
U1q(0.333211827459078*pi,1.1849946999322007*pi) q[7];
U1q(0.451079797499896*pi,0.7119631413571987*pi) q[8];
U1q(0.432965131845038*pi,1.5784863684399006*pi) q[9];
U1q(0.724206643743949*pi,1.0245034354933011*pi) q[10];
U1q(0.468542786132064*pi,1.5169664202534996*pi) q[11];
U1q(0.110410219766446*pi,0.8607264256247014*pi) q[12];
U1q(0.684097229421523*pi,0.0161296964418014*pi) q[13];
U1q(0.365802607082992*pi,1.0990446995477008*pi) q[14];
U1q(0.343098316966025*pi,1.9131028007263993*pi) q[15];
U1q(0.272883380277035*pi,1.6960141697722015*pi) q[16];
U1q(0.0647465396039917*pi,0.16662359599040144*pi) q[17];
U1q(0.548425762050301*pi,1.1153313897745*pi) q[18];
U1q(0.587566047599937*pi,1.6869153244823991*pi) q[19];
U1q(0.508654442829185*pi,1.3789039756140014*pi) q[20];
U1q(0.426143516022507*pi,0.29108178161810017*pi) q[21];
U1q(0.372420353042946*pi,0.6102801909622002*pi) q[22];
U1q(0.563166546842543*pi,1.4736835748999013*pi) q[23];
U1q(0.841178610316023*pi,0.44324779449570073*pi) q[24];
U1q(0.595020840484568*pi,0.13820845824680106*pi) q[25];
U1q(0.516148913453238*pi,1.1762598887160998*pi) q[26];
U1q(0.594607060206952*pi,0.5041657223671017*pi) q[27];
U1q(0.237036384123197*pi,0.19250687349299866*pi) q[28];
U1q(0.48545168363857*pi,0.37426548049339914*pi) q[29];
U1q(0.834699235017062*pi,0.8696463645757007*pi) q[30];
U1q(0.580748354434179*pi,0.8049113440320994*pi) q[31];
U1q(0.389330475500677*pi,1.6334544771608002*pi) q[32];
U1q(0.706241340608665*pi,0.6704237589217001*pi) q[33];
U1q(0.10594987387841*pi,0.6960347345338995*pi) q[34];
U1q(0.705312083200882*pi,0.8640075660187989*pi) q[35];
U1q(0.419364794483079*pi,0.09304470863460068*pi) q[36];
U1q(0.48738224549585*pi,1.8301776143383002*pi) q[37];
U1q(0.26628112135261*pi,0.6864778326186993*pi) q[38];
U1q(0.165148677629965*pi,1.8941906355470017*pi) q[39];
RZZ(0.5*pi) q[16],q[0];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[38];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[34];
RZZ(0.5*pi) q[6],q[30];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[31],q[20];
RZZ(0.5*pi) q[21],q[36];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[29],q[39];
RZZ(0.5*pi) q[32],q[37];
U1q(0.481596332084293*pi,0.619427399281399*pi) q[0];
U1q(0.180667240641542*pi,0.6577335481047015*pi) q[1];
U1q(0.377957699694918*pi,0.6810753828460996*pi) q[2];
U1q(0.902603863172721*pi,1.8376000830753014*pi) q[3];
U1q(0.319977446069024*pi,1.9318071113981006*pi) q[4];
U1q(0.159775955479529*pi,0.1415029018694014*pi) q[5];
U1q(0.570982917632255*pi,1.8504634461258007*pi) q[6];
U1q(0.382274736440758*pi,1.6598269231027984*pi) q[7];
U1q(0.411374932965931*pi,0.4167914644873001*pi) q[8];
U1q(0.367391392410893*pi,1.5902309504466992*pi) q[9];
U1q(0.690566857939591*pi,0.8966726996524983*pi) q[10];
U1q(0.915622860891402*pi,0.5370926665599001*pi) q[11];
U1q(0.470426496932438*pi,1.039347315732499*pi) q[12];
U1q(0.77469398896864*pi,0.2504089621150989*pi) q[13];
U1q(0.427614106279967*pi,1.676357340896601*pi) q[14];
U1q(0.499897360860844*pi,0.7064847473998981*pi) q[15];
U1q(0.485709986077636*pi,0.9853480363261014*pi) q[16];
U1q(0.498977300593643*pi,0.2745999430735999*pi) q[17];
U1q(0.887105522671993*pi,0.8555975997510998*pi) q[18];
U1q(0.655119603637497*pi,1.9773337009028005*pi) q[19];
U1q(0.393531442939411*pi,0.4476263079947991*pi) q[20];
U1q(0.61797830116403*pi,0.23135708118349996*pi) q[21];
U1q(0.549923889913638*pi,0.020639429945099153*pi) q[22];
U1q(0.786273373172207*pi,1.8274464493335998*pi) q[23];
U1q(0.530424675662512*pi,0.8698479620576016*pi) q[24];
U1q(0.362329327316798*pi,0.7643064097743988*pi) q[25];
U1q(0.436613455037825*pi,1.5528375366457006*pi) q[26];
U1q(0.489586327725317*pi,0.3141875541615988*pi) q[27];
U1q(0.344562230790869*pi,1.158575523619401*pi) q[28];
U1q(0.260190639730835*pi,0.9856847661814996*pi) q[29];
U1q(0.358913563637187*pi,1.0478580982601997*pi) q[30];
U1q(0.729772381770227*pi,1.8400000453729994*pi) q[31];
U1q(0.696617656815354*pi,1.2983588134733992*pi) q[32];
U1q(0.659555517199966*pi,0.8369445593557998*pi) q[33];
U1q(0.5348988849211*pi,1.7427905989860015*pi) q[34];
U1q(0.608706646524649*pi,0.7294300071446003*pi) q[35];
U1q(0.39881016930912*pi,1.3229179840731007*pi) q[36];
U1q(0.322454712992755*pi,0.8741582637442988*pi) q[37];
U1q(0.576005401140043*pi,0.720857333187201*pi) q[38];
U1q(0.725840805538032*pi,0.5022399655583989*pi) q[39];
rz(1.8148163982682988*pi) q[0];
rz(2.478932425627601*pi) q[1];
rz(2.511610930623899*pi) q[2];
rz(3.5788834893708987*pi) q[3];
rz(2.151496661492999*pi) q[4];
rz(1.195078717413999*pi) q[5];
rz(3.283784789749099*pi) q[6];
rz(0.7582896936511005*pi) q[7];
rz(2.1237556232700996*pi) q[8];
rz(1.5772452729179989*pi) q[9];
rz(1.8724028234739016*pi) q[10];
rz(1.4497283641531986*pi) q[11];
rz(2.973166923614201*pi) q[12];
rz(1.2191667822205012*pi) q[13];
rz(0.7835484899550984*pi) q[14];
rz(1.1562547106786027*pi) q[15];
rz(3.9147823157382007*pi) q[16];
rz(2.2119434518356016*pi) q[17];
rz(1.4751852754072985*pi) q[18];
rz(3.422387012531299*pi) q[19];
rz(0.6517590381379001*pi) q[20];
rz(2.8394232923009*pi) q[21];
rz(0.11216962948860143*pi) q[22];
rz(3.5700061319899987*pi) q[23];
rz(0.5842984463589005*pi) q[24];
rz(1.6942866724179986*pi) q[25];
rz(0.6210417911566992*pi) q[26];
rz(3.899046591554601*pi) q[27];
rz(0.6427179358685002*pi) q[28];
rz(3.6569274710396016*pi) q[29];
rz(0.5831328926179005*pi) q[30];
rz(3.760077426432499*pi) q[31];
rz(0.7089298831951005*pi) q[32];
rz(2.8521580715975006*pi) q[33];
rz(3.0617449237185*pi) q[34];
rz(1.7132387796445983*pi) q[35];
rz(0.8039572060447*pi) q[36];
rz(1.5078221082429017*pi) q[37];
rz(2.9851297829625985*pi) q[38];
rz(0.6688601568318013*pi) q[39];
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
