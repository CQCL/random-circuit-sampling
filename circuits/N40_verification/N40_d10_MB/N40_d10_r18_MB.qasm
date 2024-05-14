OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.547038392515275*pi,1.07065148064594*pi) q[0];
U1q(1.67424626591441*pi,0.36590623084964324*pi) q[1];
U1q(0.305604731988816*pi,0.865026460899438*pi) q[2];
U1q(0.317267027916368*pi,0.550918728461993*pi) q[3];
U1q(1.81710526393135*pi,0.17287609200567414*pi) q[4];
U1q(0.84133571030216*pi,1.9471449630363793*pi) q[5];
U1q(0.472709220450853*pi,0.76813975153526*pi) q[6];
U1q(0.433435046409605*pi,0.767304904441096*pi) q[7];
U1q(0.618120487432627*pi,0.636130913709496*pi) q[8];
U1q(1.05881654710576*pi,0.5833850382172362*pi) q[9];
U1q(1.1782434873184*pi,1.470573630517356*pi) q[10];
U1q(0.716684997581877*pi,1.4077294886792*pi) q[11];
U1q(1.72176061236438*pi,0.34035497277126175*pi) q[12];
U1q(0.65784232114052*pi,0.80485415357893*pi) q[13];
U1q(0.447071855343358*pi,1.860939880820418*pi) q[14];
U1q(1.75486653753312*pi,0.6831412054214808*pi) q[15];
U1q(1.46491121084703*pi,0.702670216764694*pi) q[16];
U1q(0.562554924985336*pi,1.8155251355902289*pi) q[17];
U1q(0.125654612880146*pi,1.266073829137194*pi) q[18];
U1q(0.302990404069317*pi,0.946485764990124*pi) q[19];
U1q(0.293551274338372*pi,0.355418193854948*pi) q[20];
U1q(1.5851016665728*pi,0.5597908710174079*pi) q[21];
U1q(3.620847323039533*pi,0.4551104718695842*pi) q[22];
U1q(0.664105964695536*pi,1.639163439785292*pi) q[23];
U1q(0.257381526020856*pi,0.959180919862188*pi) q[24];
U1q(1.47876926381525*pi,1.045338784063601*pi) q[25];
U1q(3.850798523208152*pi,0.7326516937064301*pi) q[26];
U1q(1.29065824444223*pi,0.3580447989890393*pi) q[27];
U1q(1.77028050906631*pi,1.2205066881542965*pi) q[28];
U1q(1.74601123374329*pi,1.237819733132623*pi) q[29];
U1q(0.279444399284685*pi,1.692612479553828*pi) q[30];
U1q(0.256370669839699*pi,1.436655273979624*pi) q[31];
U1q(0.15118666107878*pi,0.103973419009957*pi) q[32];
U1q(1.60019262153308*pi,1.2083783994275954*pi) q[33];
U1q(1.51420311354107*pi,0.2593269666909225*pi) q[34];
U1q(0.552460729143707*pi,0.0172275661668781*pi) q[35];
U1q(1.33681101206451*pi,0.7849083945073915*pi) q[36];
U1q(0.664092047796135*pi,1.011998382329422*pi) q[37];
U1q(1.204131081026*pi,1.6679316041068777*pi) q[38];
U1q(0.404724462576549*pi,0.3093612290463*pi) q[39];
RZZ(0.5*pi) q[0],q[32];
RZZ(0.5*pi) q[1],q[30];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[35],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[25],q[8];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[12],q[28];
RZZ(0.5*pi) q[39],q[13];
RZZ(0.5*pi) q[29],q[14];
RZZ(0.5*pi) q[31],q[15];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[19],q[37];
RZZ(0.5*pi) q[24],q[23];
RZZ(0.5*pi) q[26],q[36];
U1q(0.0481446129358769*pi,0.19209309857830004*pi) q[0];
U1q(0.66324127108308*pi,1.6372341044396732*pi) q[1];
U1q(0.871092299206802*pi,1.76253521515184*pi) q[2];
U1q(0.494124279814555*pi,1.7757366759256898*pi) q[3];
U1q(0.595504903554288*pi,1.9755127790283544*pi) q[4];
U1q(0.668078744278276*pi,1.6136444362892197*pi) q[5];
U1q(0.783389366996939*pi,1.9739673095746055*pi) q[6];
U1q(0.900747677884458*pi,0.32552535959174*pi) q[7];
U1q(0.47893139675157*pi,1.333254230179983*pi) q[8];
U1q(0.931723821007814*pi,1.9724564183748763*pi) q[9];
U1q(0.72576388523886*pi,1.4591437801830058*pi) q[10];
U1q(0.601946726093185*pi,0.497702646587174*pi) q[11];
U1q(0.490674227334705*pi,0.3605550896937717*pi) q[12];
U1q(0.845754877986917*pi,0.376012552187259*pi) q[13];
U1q(0.354397216956823*pi,1.8841180045873598*pi) q[14];
U1q(0.315070451972932*pi,0.08838080332718068*pi) q[15];
U1q(0.798644393501395*pi,1.242578315559654*pi) q[16];
U1q(0.717166745208919*pi,0.6100850093025598*pi) q[17];
U1q(0.265675443256605*pi,1.00132477759192*pi) q[18];
U1q(0.627438386709336*pi,0.57306204954812*pi) q[19];
U1q(0.543080672786487*pi,0.8778553622808398*pi) q[20];
U1q(0.358246592258862*pi,0.06925386182163784*pi) q[21];
U1q(0.685532785550786*pi,0.8269210563548941*pi) q[22];
U1q(0.158416094529276*pi,1.21402017529872*pi) q[23];
U1q(0.0746510791336755*pi,0.1752235036807399*pi) q[24];
U1q(0.645970283618238*pi,0.9776715931596813*pi) q[25];
U1q(0.372638484174666*pi,0.23724433651998034*pi) q[26];
U1q(0.75161643962755*pi,0.4168927918574594*pi) q[27];
U1q(0.0464678429382045*pi,0.9705095934771566*pi) q[28];
U1q(0.542756865641654*pi,0.5681523260103529*pi) q[29];
U1q(0.527355226792478*pi,1.2479567247105199*pi) q[30];
U1q(0.35554583441899*pi,1.37026063513105*pi) q[31];
U1q(0.617479812250613*pi,1.6776006112404*pi) q[32];
U1q(0.272347375276023*pi,0.8974072699406852*pi) q[33];
U1q(0.271252941038354*pi,0.7349747431962124*pi) q[34];
U1q(0.0964290213061477*pi,0.6590816096771801*pi) q[35];
U1q(0.840523349898655*pi,1.2545588485191814*pi) q[36];
U1q(0.512675231258636*pi,0.3333518032474201*pi) q[37];
U1q(0.492537374253048*pi,1.4963156561883877*pi) q[38];
U1q(0.738332194675311*pi,1.3207540990954798*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[28],q[1];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[34],q[4];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[9],q[36];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[33],q[15];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[38],q[18];
RZZ(0.5*pi) q[21],q[23];
RZZ(0.5*pi) q[22],q[26];
RZZ(0.5*pi) q[35],q[30];
U1q(0.895341376038006*pi,1.2262571209354598*pi) q[0];
U1q(0.70419237381023*pi,1.0674009726199936*pi) q[1];
U1q(0.572424551018588*pi,0.0314623229884603*pi) q[2];
U1q(0.800351790164035*pi,1.5304097817069104*pi) q[3];
U1q(0.552459025145752*pi,0.300971446688004*pi) q[4];
U1q(0.671142410895055*pi,1.8493066323036693*pi) q[5];
U1q(0.377211712735471*pi,1.007042571870544*pi) q[6];
U1q(0.186659614807771*pi,0.6984552436605398*pi) q[7];
U1q(0.597424706249026*pi,0.5049619853570202*pi) q[8];
U1q(0.480854371244171*pi,1.8027395136797866*pi) q[9];
U1q(0.286436518678142*pi,0.8962532075521761*pi) q[10];
U1q(0.307303174709299*pi,1.233463327389095*pi) q[11];
U1q(0.358047270961455*pi,1.1011696514289717*pi) q[12];
U1q(0.633144080661024*pi,1.682972028951186*pi) q[13];
U1q(0.799643652133391*pi,0.013756568051900064*pi) q[14];
U1q(0.911384005754556*pi,0.8508007406502607*pi) q[15];
U1q(0.864660695702879*pi,0.999785045814944*pi) q[16];
U1q(0.548574204406119*pi,1.5081143511232904*pi) q[17];
U1q(0.518161500103326*pi,1.5247139636895097*pi) q[18];
U1q(0.803397251041587*pi,1.8823491467143603*pi) q[19];
U1q(0.371907479002508*pi,1.09640550164307*pi) q[20];
U1q(0.372939170714035*pi,1.6576105212832477*pi) q[21];
U1q(0.608260895010972*pi,1.4901569851856742*pi) q[22];
U1q(0.691539617815837*pi,1.9391172043834004*pi) q[23];
U1q(0.131093128671961*pi,1.12686880431343*pi) q[24];
U1q(0.15214213885788*pi,0.1521815184541504*pi) q[25];
U1q(0.482930194699556*pi,0.6914306859763601*pi) q[26];
U1q(0.552659247356837*pi,0.006341736072959403*pi) q[27];
U1q(0.32611875951335*pi,0.9238026740578666*pi) q[28];
U1q(0.285774243763623*pi,0.8401054095628835*pi) q[29];
U1q(0.375970168616852*pi,1.6258966674464999*pi) q[30];
U1q(0.783983470168117*pi,1.0412331813010702*pi) q[31];
U1q(0.616523770044068*pi,0.7924376902514201*pi) q[32];
U1q(0.121874573451635*pi,1.3548770242193955*pi) q[33];
U1q(0.779668257825382*pi,0.3479310061580425*pi) q[34];
U1q(0.342730745911818*pi,1.4731864482369597*pi) q[35];
U1q(0.0862958591974115*pi,0.48855776809525153*pi) q[36];
U1q(0.590660911841923*pi,1.5423615120343301*pi) q[37];
U1q(0.864914544977823*pi,1.9849206796627978*pi) q[38];
U1q(0.545469273671138*pi,1.9166170094002801*pi) q[39];
RZZ(0.5*pi) q[0],q[35];
RZZ(0.5*pi) q[38],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[5],q[3];
RZZ(0.5*pi) q[22],q[4];
RZZ(0.5*pi) q[12],q[6];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[21],q[8];
RZZ(0.5*pi) q[29],q[9];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[11],q[15];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[16],q[14];
RZZ(0.5*pi) q[17],q[24];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[31],q[20];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[25],q[37];
RZZ(0.5*pi) q[28],q[36];
RZZ(0.5*pi) q[33],q[30];
U1q(0.183358627847108*pi,1.0702989523481499*pi) q[0];
U1q(0.348144645239013*pi,1.790605032380343*pi) q[1];
U1q(0.17389805460651*pi,1.0699809882177203*pi) q[2];
U1q(0.680222828974703*pi,1.7794564454156099*pi) q[3];
U1q(0.697587264923229*pi,0.1785395250377344*pi) q[4];
U1q(0.322359451752208*pi,0.14782970185239996*pi) q[5];
U1q(0.326118735466159*pi,1.2596642779990601*pi) q[6];
U1q(0.502359671273013*pi,0.7974673931489704*pi) q[7];
U1q(0.706449625462202*pi,0.7140670979580603*pi) q[8];
U1q(0.337695617951898*pi,1.9968569737444746*pi) q[9];
U1q(0.362281604171108*pi,0.5403359524564664*pi) q[10];
U1q(0.346474042967337*pi,1.23287581659068*pi) q[11];
U1q(0.584544087461048*pi,1.5185932535497422*pi) q[12];
U1q(0.342943007531444*pi,1.9240439737189599*pi) q[13];
U1q(0.472527615455407*pi,1.6574257464563997*pi) q[14];
U1q(0.036551292409319*pi,1.5522940968706305*pi) q[15];
U1q(0.766525387834432*pi,0.8818877871233832*pi) q[16];
U1q(0.153817746486096*pi,0.12430784166615982*pi) q[17];
U1q(0.861914238119322*pi,1.9728065485056998*pi) q[18];
U1q(0.571025064714144*pi,1.17111640409063*pi) q[19];
U1q(0.603983654719313*pi,0.2045427618279403*pi) q[20];
U1q(0.273195651550293*pi,1.7593778407014877*pi) q[21];
U1q(0.411432130871409*pi,0.17638137534067377*pi) q[22];
U1q(0.531897392793315*pi,0.33755628725104003*pi) q[23];
U1q(0.493818596433979*pi,0.061104250440160435*pi) q[24];
U1q(0.696420128328697*pi,0.19394925941326058*pi) q[25];
U1q(0.678059550059853*pi,1.0655361879049101*pi) q[26];
U1q(0.648332763316881*pi,1.8909179760849488*pi) q[27];
U1q(0.365816673970792*pi,1.5765167873790569*pi) q[28];
U1q(0.27530891493003*pi,0.7852787887376129*pi) q[29];
U1q(0.511931255071785*pi,1.0236106556580893*pi) q[30];
U1q(0.349830906079086*pi,1.7333368399523499*pi) q[31];
U1q(0.390836502290578*pi,0.5935269765659204*pi) q[32];
U1q(0.419894218701001*pi,0.5217120364540957*pi) q[33];
U1q(0.157023807645734*pi,1.2645893972202922*pi) q[34];
U1q(0.619867969635115*pi,0.00012592624675011876*pi) q[35];
U1q(0.76147183075275*pi,0.3257867598898012*pi) q[36];
U1q(0.174692840838619*pi,0.45280720904225014*pi) q[37];
U1q(0.6870535720068*pi,1.4785678397707578*pi) q[38];
U1q(0.733875585392558*pi,1.4184854911419604*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[29],q[3];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[7],q[31];
RZZ(0.5*pi) q[20],q[8];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[12],q[32];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[27],q[15];
RZZ(0.5*pi) q[23],q[18];
RZZ(0.5*pi) q[39],q[21];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[34],q[26];
RZZ(0.5*pi) q[33],q[38];
U1q(0.549211350985196*pi,1.7509938228127506*pi) q[0];
U1q(0.355717386489815*pi,0.5948128506297223*pi) q[1];
U1q(0.379271511350669*pi,1.7950966382753002*pi) q[2];
U1q(0.245724414365985*pi,0.5890546045109497*pi) q[3];
U1q(0.691705812507468*pi,0.6311495696227736*pi) q[4];
U1q(0.471021854781327*pi,1.3739739536532003*pi) q[5];
U1q(0.84473016247572*pi,0.7706736350715397*pi) q[6];
U1q(0.636674017060459*pi,1.6067316014921893*pi) q[7];
U1q(0.352968768082686*pi,0.5282824159952995*pi) q[8];
U1q(0.317831721024886*pi,0.456122730490236*pi) q[9];
U1q(0.249444545468708*pi,0.30315262422727507*pi) q[10];
U1q(0.601730379400829*pi,0.048927751828789834*pi) q[11];
U1q(0.784289300769286*pi,0.9780719656114627*pi) q[12];
U1q(0.373349816369877*pi,1.6714195242923102*pi) q[13];
U1q(0.419765728426612*pi,1.0672656061302295*pi) q[14];
U1q(0.826139677918492*pi,0.6974278958288105*pi) q[15];
U1q(0.490513605255701*pi,1.2253350674514554*pi) q[16];
U1q(0.153470397901128*pi,0.7641914763032407*pi) q[17];
U1q(0.447174384025894*pi,1.4409623669590008*pi) q[18];
U1q(0.304472049067772*pi,0.6650198707068196*pi) q[19];
U1q(0.799087244152157*pi,0.4068455657965*pi) q[20];
U1q(0.166698330591323*pi,0.7683547377425572*pi) q[21];
U1q(0.191770471938484*pi,0.016631619723934676*pi) q[22];
U1q(0.822304775323293*pi,0.7730925048740005*pi) q[23];
U1q(0.525106929735763*pi,1.5557748792857398*pi) q[24];
U1q(0.290164886630267*pi,0.4695324999815007*pi) q[25];
U1q(0.3981558744354*pi,1.06415922142906*pi) q[26];
U1q(0.682155946117693*pi,1.8801147438178294*pi) q[27];
U1q(0.350813297738209*pi,0.7396982926751061*pi) q[28];
U1q(0.13365227653664*pi,0.23591650567868339*pi) q[29];
U1q(0.325251059668046*pi,0.21101199938929938*pi) q[30];
U1q(0.583239446254837*pi,1.8402277296200094*pi) q[31];
U1q(0.54294085496009*pi,0.70370922727151*pi) q[32];
U1q(0.460773168529665*pi,1.655647402234015*pi) q[33];
U1q(0.790875125970212*pi,1.9238680284539633*pi) q[34];
U1q(0.677611508236751*pi,0.9934296945313008*pi) q[35];
U1q(0.407326590292292*pi,1.2181260616429617*pi) q[36];
U1q(0.246336195958206*pi,1.2666267731506995*pi) q[37];
U1q(0.799632649704358*pi,0.023165513707077423*pi) q[38];
U1q(0.496101897849788*pi,1.7940104500464091*pi) q[39];
RZZ(0.5*pi) q[0],q[7];
RZZ(0.5*pi) q[39],q[1];
RZZ(0.5*pi) q[2],q[33];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[38],q[6];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[19],q[20];
RZZ(0.5*pi) q[25],q[27];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[28],q[32];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[34],q[31];
U1q(0.702591628898483*pi,0.4570498171244992*pi) q[0];
U1q(0.0634398210039048*pi,1.1741511129823436*pi) q[1];
U1q(0.193370585217151*pi,0.42630348283839936*pi) q[2];
U1q(0.823524097905228*pi,1.3605350941514995*pi) q[3];
U1q(0.592995021507371*pi,1.0176194484616747*pi) q[4];
U1q(0.611713539819278*pi,0.6019911469877002*pi) q[5];
U1q(0.705590477624459*pi,1.2724702727592199*pi) q[6];
U1q(0.319071817581626*pi,1.9672466457622004*pi) q[7];
U1q(0.519308422683425*pi,0.8409607688130993*pi) q[8];
U1q(0.299627140545323*pi,0.8881263745011356*pi) q[9];
U1q(0.488275159967013*pi,1.100926255921756*pi) q[10];
U1q(0.452340034646896*pi,0.5548051246094801*pi) q[11];
U1q(0.480743701569188*pi,1.2086315381714616*pi) q[12];
U1q(0.682353207444497*pi,0.1777347164169596*pi) q[13];
U1q(0.651945866754196*pi,1.3652941755912007*pi) q[14];
U1q(0.709303865666896*pi,0.6397686591568821*pi) q[15];
U1q(0.182568442635252*pi,1.4182956599232952*pi) q[16];
U1q(0.400197070395126*pi,1.3922356445260995*pi) q[17];
U1q(0.128737338589921*pi,1.3203538265100008*pi) q[18];
U1q(0.923316740857806*pi,0.4250640395299303*pi) q[19];
U1q(0.685172720909719*pi,1.5970947589163007*pi) q[20];
U1q(0.207033544468966*pi,1.4761633171524071*pi) q[21];
U1q(0.316951843913045*pi,1.7368999517766852*pi) q[22];
U1q(0.151770566051464*pi,1.9771756187954992*pi) q[23];
U1q(0.448673694121899*pi,0.44393461014963087*pi) q[24];
U1q(0.764829229740371*pi,0.2989927787407005*pi) q[25];
U1q(0.3553993101304*pi,0.83782660783843*pi) q[26];
U1q(0.412584842823267*pi,0.748612443686639*pi) q[27];
U1q(0.238668981522537*pi,1.2762690799666974*pi) q[28];
U1q(0.976175167087191*pi,1.278937072882453*pi) q[29];
U1q(0.492200121404207*pi,1.1174146647417*pi) q[30];
U1q(0.68363895659861*pi,1.3016732233572998*pi) q[31];
U1q(0.667170243104969*pi,0.32773051680525*pi) q[32];
U1q(0.785536226335058*pi,0.625310940479995*pi) q[33];
U1q(0.66379649932007*pi,1.9476456245913223*pi) q[34];
U1q(0.67032996614356*pi,0.6348416954112999*pi) q[35];
U1q(0.0825305310747628*pi,1.3586031885303917*pi) q[36];
U1q(0.611544491496616*pi,1.715722250352*pi) q[37];
U1q(0.439300896127917*pi,0.9485880942485974*pi) q[38];
U1q(0.354467168682206*pi,1.2562604498873*pi) q[39];
rz(3.780419665257*pi) q[0];
rz(0.19572377363445526*pi) q[1];
rz(2.771366901278199*pi) q[2];
rz(2.642141346205401*pi) q[3];
rz(1.7153868490919848*pi) q[4];
rz(0.36479187397269897*pi) q[5];
rz(0.12873057358203965*pi) q[6];
rz(2.0693927437464*pi) q[7];
rz(1.3247026698210007*pi) q[8];
rz(3.944369955526165*pi) q[9];
rz(1.6511035433019448*pi) q[10];
rz(1.83395964108799*pi) q[11];
rz(3.680230895108039*pi) q[12];
rz(3.3578434954948797*pi) q[13];
rz(3.0754843621456995*pi) q[14];
rz(1.5431397164349185*pi) q[15];
rz(2.033451929498405*pi) q[16];
rz(3.7561118877066004*pi) q[17];
rz(1.3815626264591003*pi) q[18];
rz(0.3007472949945207*pi) q[19];
rz(2.1009128024606003*pi) q[20];
rz(1.6942480152040922*pi) q[21];
rz(2.058187522169616*pi) q[22];
rz(1.2927500451029008*pi) q[23];
rz(2.1184667176892003*pi) q[24];
rz(3.405497497738599*pi) q[25];
rz(2.5704336633285703*pi) q[26];
rz(3.291306669152311*pi) q[27];
rz(1.7428644382324041*pi) q[28];
rz(3.620788199782478*pi) q[29];
rz(2.7789578250814007*pi) q[30];
rz(3.3415440842849993*pi) q[31];
rz(2.1521602497629004*pi) q[32];
rz(2.184016872285305*pi) q[33];
rz(2.457755634920378*pi) q[34];
rz(1.7016011213182995*pi) q[35];
rz(2.5604060699236086*pi) q[36];
rz(3.1258152810107003*pi) q[37];
rz(3.6784832795255227*pi) q[38];
rz(2.5562652610915*pi) q[39];
U1q(0.702591628898483*pi,1.237469482381494*pi) q[0];
U1q(0.0634398210039048*pi,0.369874886616737*pi) q[1];
U1q(0.193370585217151*pi,0.197670384116645*pi) q[2];
U1q(0.823524097905228*pi,1.002676440356888*pi) q[3];
U1q(1.59299502150737*pi,1.7330062975536609*pi) q[4];
U1q(0.611713539819278*pi,1.9667830209604553*pi) q[5];
U1q(0.705590477624459*pi,0.40120084634126*pi) q[6];
U1q(0.319071817581626*pi,1.036639389508651*pi) q[7];
U1q(3.519308422683425*pi,1.165663438634081*pi) q[8];
U1q(0.299627140545323*pi,1.832496330027327*pi) q[9];
U1q(3.488275159967014*pi,1.752029799223693*pi) q[10];
U1q(1.4523400346469*pi,1.3887647656974709*pi) q[11];
U1q(0.480743701569188*pi,1.888862433279414*pi) q[12];
U1q(1.6823532074445*pi,0.535578211911841*pi) q[13];
U1q(1.6519458667542*pi,1.4407785377369*pi) q[14];
U1q(0.709303865666896*pi,1.18290837559179*pi) q[15];
U1q(1.18256844263525*pi,0.451747589421648*pi) q[16];
U1q(0.400197070395126*pi,0.148347532232703*pi) q[17];
U1q(1.12873733858992*pi,1.701916452969102*pi) q[18];
U1q(0.923316740857806*pi,1.725811334524451*pi) q[19];
U1q(1.68517272090972*pi,0.698007561376847*pi) q[20];
U1q(3.207033544468965*pi,0.170411332356541*pi) q[21];
U1q(1.31695184391304*pi,0.7950874739462901*pi) q[22];
U1q(0.151770566051464*pi,0.269925663898413*pi) q[23];
U1q(0.448673694121899*pi,1.562401327838858*pi) q[24];
U1q(1.76482922974037*pi,0.704490276479294*pi) q[25];
U1q(0.3553993101304*pi,0.408260271167068*pi) q[26];
U1q(1.41258484282327*pi,1.039919112838947*pi) q[27];
U1q(1.23866898152254*pi,0.0191335181991048*pi) q[28];
U1q(1.97617516708719*pi,1.89972527266496*pi) q[29];
U1q(0.492200121404207*pi,0.8963724898230401*pi) q[30];
U1q(0.68363895659861*pi,1.643217307642281*pi) q[31];
U1q(1.66717024310497*pi,1.4798907665682*pi) q[32];
U1q(0.785536226335058*pi,1.809327812765264*pi) q[33];
U1q(1.66379649932007*pi,1.4054012595117369*pi) q[34];
U1q(0.67032996614356*pi,1.336442816729595*pi) q[35];
U1q(0.0825305310747628*pi,0.91900925845397*pi) q[36];
U1q(1.61154449149662*pi,1.841537531362648*pi) q[37];
U1q(3.439300896127917*pi,1.627071373774117*pi) q[38];
U1q(0.354467168682206*pi,0.812525710978756*pi) q[39];
RZZ(0.5*pi) q[0],q[7];
RZZ(0.5*pi) q[39],q[1];
RZZ(0.5*pi) q[2],q[33];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[38],q[6];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[19],q[20];
RZZ(0.5*pi) q[25],q[27];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[28],q[32];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[34],q[31];
U1q(1.5492113509852*pi,1.5314134880697199*pi) q[0];
U1q(0.355717386489815*pi,0.790536624264138*pi) q[1];
U1q(1.37927151135067*pi,0.56646353955349*pi) q[2];
U1q(1.24572441436599*pi,1.2311959507163102*pi) q[3];
U1q(1.69170581250747*pi,0.11947617639256874*pi) q[4];
U1q(3.471021854781327*pi,1.7387658276258802*pi) q[5];
U1q(0.84473016247572*pi,1.899404208653577*pi) q[6];
U1q(0.636674017060459*pi,1.67612434523862*pi) q[7];
U1q(3.352968768082686*pi,1.4783417914518362*pi) q[8];
U1q(0.317831721024886*pi,1.40049268601636*pi) q[9];
U1q(3.750555454531292*pi,1.5498034309181725*pi) q[10];
U1q(3.398269620599171*pi,1.894642138478161*pi) q[11];
U1q(1.78428930076929*pi,1.65830286071941*pi) q[12];
U1q(1.37334981636988*pi,0.04189340403648939*pi) q[13];
U1q(1.41976572842661*pi,0.7388071071978579*pi) q[14];
U1q(3.826139677918492*pi,1.24056761226377*pi) q[15];
U1q(3.509486394744299*pi,0.6447081818934415*pi) q[16];
U1q(0.153470397901128*pi,1.5203033640098802*pi) q[17];
U1q(3.447174384025894*pi,1.5813079125201777*pi) q[18];
U1q(1.30447204906777*pi,1.9657671657013398*pi) q[19];
U1q(1.79908724415216*pi,1.8882567544966387*pi) q[20];
U1q(3.833301669408678*pi,0.8782199117664353*pi) q[21];
U1q(3.191770471938484*pi,0.5153558059990258*pi) q[22];
U1q(1.82230477532329*pi,0.06584254997687*pi) q[23];
U1q(0.525106929735763*pi,0.674241596974968*pi) q[24];
U1q(3.709835113369732*pi,0.5339505552385172*pi) q[25];
U1q(1.3981558744354*pi,0.6345928847576601*pi) q[26];
U1q(1.68215594611769*pi,0.9084168127077636*pi) q[27];
U1q(1.35081329773821*pi,1.5557043054907393*pi) q[28];
U1q(3.86634772346336*pi,0.9427458398687243*pi) q[29];
U1q(0.325251059668046*pi,0.9899698244706601*pi) q[30];
U1q(1.58323944625484*pi,0.18177181390499997*pi) q[31];
U1q(3.457059145039909*pi,1.1039120561019418*pi) q[32];
U1q(0.460773168529665*pi,1.8396642745193201*pi) q[33];
U1q(1.79087512597021*pi,1.4291788556491447*pi) q[34];
U1q(1.67761150823675*pi,0.6950308158495502*pi) q[35];
U1q(1.40732659029229*pi,1.77853213156653*pi) q[36];
U1q(3.246336195958206*pi,0.2906330085639415*pi) q[37];
U1q(1.79963264970436*pi,0.5524939543156346*pi) q[38];
U1q(0.496101897849788*pi,0.350275711137884*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[29],q[3];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[7],q[31];
RZZ(0.5*pi) q[20],q[8];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[12],q[32];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[27],q[15];
RZZ(0.5*pi) q[23],q[18];
RZZ(0.5*pi) q[39],q[21];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[34],q[26];
RZZ(0.5*pi) q[33],q[38];
U1q(1.18335862784711*pi,1.2121083585343189*pi) q[0];
U1q(0.348144645239013*pi,0.9863288060147599*pi) q[1];
U1q(3.17389805460651*pi,0.2915791896110347*pi) q[2];
U1q(1.6802228289747*pi,1.040794109811653*pi) q[3];
U1q(0.697587264923229*pi,1.6668661318075277*pi) q[4];
U1q(3.677640548247792*pi,0.9649100794266092*pi) q[5];
U1q(0.326118735466159*pi,1.388394851581096*pi) q[6];
U1q(0.502359671273013*pi,0.8668601368954001*pi) q[7];
U1q(0.706449625462202*pi,0.6641264734145951*pi) q[8];
U1q(0.337695617951898*pi,1.94122692927062*pi) q[9];
U1q(1.36228160417111*pi,0.31262010268897966*pi) q[10];
U1q(3.653525957032663*pi,0.7106940737162715*pi) q[11];
U1q(3.4154559125389508*pi,1.1177815727810763*pi) q[12];
U1q(0.342943007531444*pi,1.2945178534631383*pi) q[13];
U1q(0.472527615455407*pi,1.32896724752403*pi) q[14];
U1q(1.03655129240932*pi,0.3857014112219491*pi) q[15];
U1q(1.76652538783443*pi,0.9881554622215118*pi) q[16];
U1q(1.1538177464861*pi,1.8804197293728002*pi) q[17];
U1q(0.861914238119322*pi,0.11315209406686068*pi) q[18];
U1q(1.57102506471414*pi,1.4596706323175264*pi) q[19];
U1q(0.603983654719313*pi,0.6859539505280816*pi) q[20];
U1q(3.273195651550293*pi,0.8871968088075022*pi) q[21];
U1q(3.411432130871409*pi,0.6751055616157657*pi) q[22];
U1q(3.468102607206685*pi,1.5013787675998178*pi) q[23];
U1q(1.49381859643398*pi,1.17957096812938*pi) q[24];
U1q(1.6964201283287*pi,1.8095337958067454*pi) q[25];
U1q(1.67805955005985*pi,0.6332159182818158*pi) q[26];
U1q(0.648332763316881*pi,0.9192200449748862*pi) q[27];
U1q(1.36581667397079*pi,1.3925228001946843*pi) q[28];
U1q(3.72469108506997*pi,1.3933835568097983*pi) q[29];
U1q(0.511931255071785*pi,1.8025684807394802*pi) q[30];
U1q(3.349830906079086*pi,0.288662703572667*pi) q[31];
U1q(1.39083650229058*pi,1.2140943068075352*pi) q[32];
U1q(0.419894218701001*pi,1.7057289087393999*pi) q[33];
U1q(0.157023807645734*pi,0.7699002244154758*pi) q[34];
U1q(1.61986796963512*pi,1.688334584134087*pi) q[35];
U1q(1.76147183075275*pi,0.670871433319689*pi) q[36];
U1q(0.174692840838619*pi,1.4768134444555083*pi) q[37];
U1q(0.6870535720068*pi,1.0078962803793075*pi) q[38];
U1q(0.733875585392558*pi,1.974750752233432*pi) q[39];
RZZ(0.5*pi) q[0],q[35];
RZZ(0.5*pi) q[38],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[5],q[3];
RZZ(0.5*pi) q[22],q[4];
RZZ(0.5*pi) q[12],q[6];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[21],q[8];
RZZ(0.5*pi) q[29],q[9];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[11],q[15];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[16],q[14];
RZZ(0.5*pi) q[17],q[24];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[31],q[20];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[25],q[37];
RZZ(0.5*pi) q[28],q[36];
RZZ(0.5*pi) q[33],q[30];
U1q(1.89534137603801*pi,0.368066527121619*pi) q[0];
U1q(0.70419237381023*pi,1.2631247462544102*pi) q[1];
U1q(1.57242455101859*pi,0.25306052438177673*pi) q[2];
U1q(3.800351790164036*pi,0.7917474461029634*pi) q[3];
U1q(0.552459025145752*pi,1.789298053457808*pi) q[4];
U1q(3.328857589104945*pi,1.2634331489753692*pi) q[5];
U1q(0.377211712735471*pi,1.135773145452581*pi) q[6];
U1q(0.186659614807771*pi,1.7678479874069701*pi) q[7];
U1q(0.597424706249026*pi,1.4550213608135554*pi) q[8];
U1q(1.48085437124417*pi,1.74710946920594*pi) q[9];
U1q(0.286436518678142*pi,1.6685373577846896*pi) q[10];
U1q(1.3073031747093*pi,1.710106562917865*pi) q[11];
U1q(1.35804727096146*pi,0.535205174901848*pi) q[12];
U1q(0.633144080661024*pi,0.05344590869535848*pi) q[13];
U1q(3.799643652133392*pi,0.6852980691195398*pi) q[14];
U1q(1.91138400575456*pi,1.6842080550015779*pi) q[15];
U1q(1.86466069570288*pi,0.10605272091306261*pi) q[16];
U1q(1.54857420440612*pi,1.4966132199156652*pi) q[17];
U1q(1.51816150010333*pi,0.6650595092507057*pi) q[18];
U1q(0.803397251041587*pi,1.1709033749412567*pi) q[19];
U1q(1.37190747900251*pi,0.5778166903432114*pi) q[20];
U1q(0.372939170714035*pi,1.785429489389272*pi) q[21];
U1q(1.60826089501097*pi,1.3613299517707746*pi) q[22];
U1q(1.69153961781584*pi,0.8998178504674521*pi) q[23];
U1q(1.13109312867196*pi,1.1138064142561093*pi) q[24];
U1q(0.15214213885788*pi,0.7677660548476395*pi) q[25];
U1q(0.482930194699556*pi,0.2591104163532756*pi) q[26];
U1q(0.552659247356837*pi,0.03464380496289632*pi) q[27];
U1q(3.673881240486649*pi,1.0452369135158839*pi) q[28];
U1q(3.714225756236377*pi,1.3385569359845313*pi) q[29];
U1q(3.3759701686168517*pi,1.4048544925278996*pi) q[30];
U1q(0.783983470168117*pi,0.5965590449213969*pi) q[31];
U1q(3.616523770044068*pi,0.4130050204930349*pi) q[32];
U1q(0.121874573451635*pi,1.5388938965046899*pi) q[33];
U1q(0.779668257825382*pi,0.8532418333532279*pi) q[34];
U1q(0.342730745911818*pi,0.1613951061243073*pi) q[35];
U1q(0.0862958591974115*pi,0.8336424415251393*pi) q[36];
U1q(1.59066091184192*pi,1.5663677474475834*pi) q[37];
U1q(1.86491454497782*pi,0.5142491202713573*pi) q[38];
U1q(3.545469273671138*pi,1.472882270491751*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[28],q[1];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[34],q[4];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[9],q[36];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[33],q[15];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[38],q[18];
RZZ(0.5*pi) q[21],q[23];
RZZ(0.5*pi) q[22],q[26];
RZZ(0.5*pi) q[35],q[30];
U1q(3.951855387064122*pi,1.402230549478781*pi) q[0];
U1q(1.66324127108308*pi,1.8329578780740898*pi) q[1];
U1q(3.128907700793198*pi,0.5219876322183961*pi) q[2];
U1q(3.5058757201854442*pi,1.5464205518841787*pi) q[3];
U1q(0.595504903554288*pi,0.4638393857981571*pi) q[4];
U1q(1.66807874427828*pi,1.4990953449898237*pi) q[5];
U1q(1.78338936699694*pi,0.10269788315664008*pi) q[6];
U1q(1.90074767788446*pi,1.3949181033381697*pi) q[7];
U1q(0.47893139675157*pi,1.2833136056365149*pi) q[8];
U1q(3.068276178992187*pi,0.5773925645108573*pi) q[9];
U1q(0.72576388523886*pi,1.2314279304155198*pi) q[10];
U1q(0.601946726093185*pi,1.9743458821159345*pi) q[11];
U1q(1.49067422733471*pi,0.7945906131666387*pi) q[12];
U1q(0.845754877986917*pi,0.7464864319314382*pi) q[13];
U1q(3.645602783043176*pi,1.8149366325840788*pi) q[14];
U1q(3.684929548027069*pi,1.4466279923246552*pi) q[15];
U1q(3.2013556064986037*pi,0.8632594511683452*pi) q[16];
U1q(3.717166745208919*pi,1.598583878094935*pi) q[17];
U1q(1.26567544325661*pi,0.18844869534829511*pi) q[18];
U1q(0.627438386709336*pi,0.8616162777750169*pi) q[19];
U1q(3.4569193272135132*pi,0.7963668297054416*pi) q[20];
U1q(0.358246592258862*pi,1.197072829927662*pi) q[21];
U1q(3.685532785550786*pi,1.6980940229400021*pi) q[22];
U1q(0.158416094529276*pi,1.1747208213827722*pi) q[23];
U1q(1.07465107913368*pi,0.16216111362341623*pi) q[24];
U1q(1.64597028361824*pi,0.5932561295531755*pi) q[25];
U1q(1.37263848417467*pi,0.8049240668968958*pi) q[26];
U1q(1.75161643962755*pi,1.4451948607473994*pi) q[27];
U1q(3.9535321570617956*pi,0.9985299940965837*pi) q[28];
U1q(3.457243134358346*pi,1.6105100195370543*pi) q[29];
U1q(3.472644773207522*pi,0.7827944352638818*pi) q[30];
U1q(3.35554583441899*pi,0.9255864987513771*pi) q[31];
U1q(1.61747981225061*pi,1.527842099504059*pi) q[32];
U1q(1.27234737527602*pi,1.0814241422259805*pi) q[33];
U1q(0.271252941038354*pi,0.24028557039139775*pi) q[34];
U1q(0.0964290213061477*pi,1.347290267564527*pi) q[35];
U1q(3.840523349898655*pi,1.599643521949079*pi) q[36];
U1q(3.487324768741364*pi,0.7753774562345006*pi) q[37];
U1q(3.507462625746952*pi,1.0028541437457692*pi) q[38];
U1q(3.261667805324689*pi,0.06874518079654424*pi) q[39];
RZZ(0.5*pi) q[0],q[32];
RZZ(0.5*pi) q[1],q[30];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[35],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[25],q[8];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[12],q[28];
RZZ(0.5*pi) q[39],q[13];
RZZ(0.5*pi) q[29],q[14];
RZZ(0.5*pi) q[31],q[15];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[19],q[37];
RZZ(0.5*pi) q[24],q[23];
RZZ(0.5*pi) q[26],q[36];
U1q(1.54703839251528*pi,1.5236721674111502*pi) q[0];
U1q(1.67424626591441*pi,0.10428575166412268*pi) q[1];
U1q(1.30560473198882*pi,0.41949638647078924*pi) q[2];
U1q(1.31726702791637*pi,0.7712384993478816*pi) q[3];
U1q(0.817105263931349*pi,1.6612026987754671*pi) q[4];
U1q(0.84133571030216*pi,1.8325958717369837*pi) q[5];
U1q(1.47270922045085*pi,0.308525441195985*pi) q[6];
U1q(1.43343504640961*pi,0.9531385584888152*pi) q[7];
U1q(0.618120487432627*pi,0.5861902891660353*pi) q[8];
U1q(1.05881654710576*pi,1.9664639446684937*pi) q[9];
U1q(0.178243487318402*pi,1.2428577807498797*pi) q[10];
U1q(0.716684997581877*pi,1.8843727242079646*pi) q[11];
U1q(1.72176061236438*pi,1.8147907300891442*pi) q[12];
U1q(0.65784232114052*pi,1.1753280333231082*pi) q[13];
U1q(1.44707185534336*pi,1.8381147563510205*pi) q[14];
U1q(1.75486653753312*pi,0.8518675902303574*pi) q[15];
U1q(3.464911210847027*pi,1.4031675499633076*pi) q[16];
U1q(1.56255492498534*pi,1.3931437518072656*pi) q[17];
U1q(0.125654612880146*pi,0.4531977468935753*pi) q[18];
U1q(0.302990404069317*pi,1.2350399932170166*pi) q[19];
U1q(1.29355127433837*pi,0.31880399813133486*pi) q[20];
U1q(0.585101666572799*pi,0.6876098391234322*pi) q[21];
U1q(1.62084732303953*pi,1.0699046074253182*pi) q[22];
U1q(0.664105964695536*pi,1.599864085869342*pi) q[23];
U1q(1.25738152602086*pi,1.3782036974419691*pi) q[24];
U1q(1.47876926381525*pi,0.5255889386492578*pi) q[25];
U1q(1.85079852320815*pi,1.309516709710448*pi) q[26];
U1q(1.29065824444223*pi,0.5040428536158217*pi) q[27];
U1q(1.77028050906631*pi,0.7485328994194451*pi) q[28];
U1q(3.746011233743287*pi,0.94084261241479*pi) q[29];
U1q(1.27944439928469*pi,0.33813868042057393*pi) q[30];
U1q(1.2563706698397*pi,1.8591918599028086*pi) q[31];
U1q(0.15118666107878*pi,1.9542149072736086*pi) q[32];
U1q(1.60019262153308*pi,1.7704530127390683*pi) q[33];
U1q(0.514203113541071*pi,1.7646377938861075*pi) q[34];
U1q(0.552460729143707*pi,1.7054362240542176*pi) q[35];
U1q(1.33681101206451*pi,1.0692939759608748*pi) q[36];
U1q(1.66409204779613*pi,1.0967308771524964*pi) q[37];
U1q(1.204131081026*pi,1.8312381958272752*pi) q[38];
U1q(1.40472446257655*pi,0.08013805084572767*pi) q[39];
rz(2.47632783258885*pi) q[0];
rz(1.8957142483358773*pi) q[1];
rz(1.5805036135292108*pi) q[2];
rz(3.2287615006521184*pi) q[3];
rz(2.338797301224533*pi) q[4];
rz(0.16740412826301632*pi) q[5];
rz(3.691474558804015*pi) q[6];
rz(1.0468614415111848*pi) q[7];
rz(1.4138097108339647*pi) q[8];
rz(2.0335360553315063*pi) q[9];
rz(0.7571422192501203*pi) q[10];
rz(2.1156272757920354*pi) q[11];
rz(2.185209269910856*pi) q[12];
rz(0.8246719666768917*pi) q[13];
rz(2.1618852436489795*pi) q[14];
rz(3.1481324097696426*pi) q[15];
rz(0.5968324500366925*pi) q[16];
rz(0.6068562481927344*pi) q[17];
rz(3.5468022531064247*pi) q[18];
rz(2.7649600067829834*pi) q[19];
rz(3.681196001868665*pi) q[20];
rz(3.312390160876568*pi) q[21];
rz(2.9300953925746818*pi) q[22];
rz(2.400135914130658*pi) q[23];
rz(2.621796302558031*pi) q[24];
rz(1.4744110613507422*pi) q[25];
rz(2.690483290289552*pi) q[26];
rz(1.4959571463841783*pi) q[27];
rz(3.251467100580555*pi) q[28];
rz(3.05915738758521*pi) q[29];
rz(1.661861319579426*pi) q[30];
rz(0.14080814009719145*pi) q[31];
rz(2.0457850927263914*pi) q[32];
rz(2.2295469872609317*pi) q[33];
rz(0.23536220611389247*pi) q[34];
rz(0.2945637759457824*pi) q[35];
rz(0.9307060240391252*pi) q[36];
rz(2.9032691228475036*pi) q[37];
rz(2.168761804172725*pi) q[38];
rz(1.9198619491542723*pi) q[39];
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
