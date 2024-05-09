OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(1.22607242943562*pi,0.2735245267262863*pi) q[0];
U1q(0.504537522341449*pi,0.986017996415763*pi) q[1];
U1q(0.457217201088832*pi,0.297453124791507*pi) q[2];
U1q(1.73597150385386*pi,1.340056363492753*pi) q[3];
U1q(0.177978091942841*pi,0.251135303320085*pi) q[4];
U1q(0.381903035888817*pi,0.57910502454181*pi) q[5];
U1q(0.285077272781546*pi,1.088671141864257*pi) q[6];
U1q(0.261083421483919*pi,1.572312561558225*pi) q[7];
U1q(0.451408816739829*pi,0.0499629944877807*pi) q[8];
U1q(1.4539161772817*pi,1.8102988523611478*pi) q[9];
U1q(0.39295029666598*pi,1.400328908750651*pi) q[10];
U1q(3.678513576370081*pi,1.1369225016930695*pi) q[11];
U1q(1.75442388410449*pi,1.72065279836091*pi) q[12];
U1q(1.60895463426798*pi,1.6949858770967325*pi) q[13];
U1q(1.79818232191593*pi,0.26955352290765056*pi) q[14];
U1q(3.112227896741101*pi,1.3502546265182698*pi) q[15];
U1q(0.725200416262468*pi,0.615776296025547*pi) q[16];
U1q(1.27818246439645*pi,1.3994747712858482*pi) q[17];
U1q(1.62954355087424*pi,1.3732115305984087*pi) q[18];
U1q(0.402552435403241*pi,0.5981148704217101*pi) q[19];
U1q(0.709339892838627*pi,0.809600494688921*pi) q[20];
U1q(0.208190733537805*pi,0.117671711891689*pi) q[21];
U1q(0.780919435302561*pi,1.4501698645694199*pi) q[22];
U1q(1.81455597917983*pi,0.5137951785139113*pi) q[23];
U1q(1.70566619095162*pi,1.4896763031337867*pi) q[24];
U1q(1.72425369600469*pi,1.969365945330718*pi) q[25];
U1q(3.729260185977075*pi,1.0732811629538124*pi) q[26];
U1q(1.48291952057387*pi,1.5464530991660042*pi) q[27];
U1q(1.15678998216674*pi,1.8508130376470135*pi) q[28];
U1q(0.213720693960634*pi,0.0620993749752161*pi) q[29];
U1q(0.464295008165292*pi,1.0252995101205*pi) q[30];
U1q(1.51191029990819*pi,1.3769449390048831*pi) q[31];
U1q(0.214978616314598*pi,0.65854024874142*pi) q[32];
U1q(1.74386542705225*pi,1.5759052903931074*pi) q[33];
U1q(0.536864950041988*pi,0.257337446946574*pi) q[34];
U1q(0.813901029743174*pi,1.783003817773365*pi) q[35];
U1q(1.71561828524044*pi,1.6686539787730195*pi) q[36];
U1q(1.4691890052904*pi,0.9091318436390488*pi) q[37];
U1q(1.62037023629113*pi,1.8997341251652116*pi) q[38];
U1q(1.20778960319836*pi,0.8843754342345801*pi) q[39];
RZZ(0.5*pi) q[0],q[13];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[23];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[16],q[10];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[12],q[24];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[37],q[15];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[21],q[31];
RZZ(0.5*pi) q[30],q[29];
RZZ(0.5*pi) q[32],q[38];
RZZ(0.5*pi) q[39],q[36];
U1q(0.373120496531788*pi,0.40167854252663626*pi) q[0];
U1q(0.246031181953476*pi,1.644676921721163*pi) q[1];
U1q(0.774451169599276*pi,0.22984282142516999*pi) q[2];
U1q(0.449941815843335*pi,1.120119629277493*pi) q[3];
U1q(0.363151151987262*pi,0.24641923359091988*pi) q[4];
U1q(0.460466540832382*pi,1.83800880618917*pi) q[5];
U1q(0.866705232522648*pi,1.9956843389857202*pi) q[6];
U1q(0.889469195401386*pi,1.41423063124192*pi) q[7];
U1q(0.279906834554469*pi,1.2228230221736802*pi) q[8];
U1q(0.170496295367881*pi,1.734161425920048*pi) q[9];
U1q(0.576880903624925*pi,1.2014782150581498*pi) q[10];
U1q(0.40900821342524*pi,1.3413108578981596*pi) q[11];
U1q(0.626067219090242*pi,0.3637497916958199*pi) q[12];
U1q(0.252245210243244*pi,1.5975436017894227*pi) q[13];
U1q(0.364637318501898*pi,1.0421908366411206*pi) q[14];
U1q(0.38232462571931*pi,0.9335576991485097*pi) q[15];
U1q(0.825073910172482*pi,1.063852902702176*pi) q[16];
U1q(0.121898006359649*pi,1.9586298819331276*pi) q[17];
U1q(0.437059336714011*pi,1.467164468570779*pi) q[18];
U1q(0.747075627737868*pi,1.33743496087595*pi) q[19];
U1q(0.256687633493107*pi,1.96970583949759*pi) q[20];
U1q(0.533134371419049*pi,0.7371042993915902*pi) q[21];
U1q(0.28655688345214*pi,1.8078733447403703*pi) q[22];
U1q(0.409043988065733*pi,0.45406552861682137*pi) q[23];
U1q(0.588862092278428*pi,0.9705497071636167*pi) q[24];
U1q(0.651432422593109*pi,0.585412989712268*pi) q[25];
U1q(0.562243832241824*pi,1.1863529246829825*pi) q[26];
U1q(0.265297618010252*pi,0.400430959843614*pi) q[27];
U1q(0.866443537943486*pi,1.8463191814457534*pi) q[28];
U1q(0.427018510665143*pi,0.77752459342244*pi) q[29];
U1q(0.723487495290866*pi,1.277185958813884*pi) q[30];
U1q(0.534445586322182*pi,0.9892263131992833*pi) q[31];
U1q(0.329636373622171*pi,0.91748008734851*pi) q[32];
U1q(0.330203750587457*pi,0.8966579323633272*pi) q[33];
U1q(0.421350307431848*pi,0.5039982164999302*pi) q[34];
U1q(0.262521031339458*pi,0.08426892468567004*pi) q[35];
U1q(0.138205472909039*pi,0.021615609101489408*pi) q[36];
U1q(0.472805077145162*pi,1.3265064392733987*pi) q[37];
U1q(0.871641025358626*pi,1.0666326727478914*pi) q[38];
U1q(0.525011386274555*pi,1.50167187660491*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[7],q[25];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[13],q[12];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[38],q[17];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[31],q[24];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[34],q[28];
RZZ(0.5*pi) q[32],q[37];
U1q(0.407684207168542*pi,1.9364850359895867*pi) q[0];
U1q(0.777402668942508*pi,1.90778743932102*pi) q[1];
U1q(0.456352103442754*pi,1.82213255116267*pi) q[2];
U1q(0.586837004093232*pi,1.966491184361363*pi) q[3];
U1q(0.558926682163639*pi,0.25723076101489006*pi) q[4];
U1q(0.865236431767738*pi,1.23533805223944*pi) q[5];
U1q(0.588961991900659*pi,1.27863954693328*pi) q[6];
U1q(0.630274028447919*pi,1.3947094470816*pi) q[7];
U1q(0.512182444603415*pi,0.39502696808780957*pi) q[8];
U1q(0.248142965611392*pi,1.1662235737740776*pi) q[9];
U1q(0.71401742281898*pi,0.5810028850696503*pi) q[10];
U1q(0.49446735226855*pi,1.9993730497477191*pi) q[11];
U1q(0.902788651046118*pi,0.9654720519171205*pi) q[12];
U1q(0.568126942093718*pi,0.9467685794105227*pi) q[13];
U1q(0.232908535575783*pi,0.14806555661631027*pi) q[14];
U1q(0.357947144507264*pi,0.5649802253610599*pi) q[15];
U1q(0.595280506566631*pi,1.4426287922904302*pi) q[16];
U1q(0.234994472341371*pi,0.8626962576097084*pi) q[17];
U1q(0.652621610214298*pi,1.0492187048786192*pi) q[18];
U1q(0.607982830358834*pi,1.6940816586851097*pi) q[19];
U1q(0.54787794365631*pi,1.0036007792721802*pi) q[20];
U1q(0.43291495723573*pi,0.1598767510633401*pi) q[21];
U1q(0.424223855428175*pi,1.3714566029944004*pi) q[22];
U1q(0.607456765577331*pi,1.4937141244207615*pi) q[23];
U1q(0.64230619945232*pi,0.676091395816047*pi) q[24];
U1q(0.701409519547642*pi,0.9879821740538879*pi) q[25];
U1q(0.889791406770676*pi,1.5387156851021926*pi) q[26];
U1q(0.699186020411433*pi,0.8494626511729839*pi) q[27];
U1q(0.88571966855043*pi,0.6562142795130836*pi) q[28];
U1q(0.937878591820396*pi,0.15911585219576008*pi) q[29];
U1q(0.281324134352789*pi,0.50762785183601*pi) q[30];
U1q(0.586624985485106*pi,0.060440035016563254*pi) q[31];
U1q(0.807941432995997*pi,0.6464140267101102*pi) q[32];
U1q(0.775822840542209*pi,0.9318184207135873*pi) q[33];
U1q(0.65910886547728*pi,1.3639534867050402*pi) q[34];
U1q(0.748192807185111*pi,1.54015372008109*pi) q[35];
U1q(0.946164591976671*pi,0.07682384963845923*pi) q[36];
U1q(0.302404815007045*pi,0.23454672639533847*pi) q[37];
U1q(0.856194648754017*pi,0.6623054993334518*pi) q[38];
U1q(0.525096637284772*pi,1.5865508414524605*pi) q[39];
RZZ(0.5*pi) q[0],q[16];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[4],q[7];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[6],q[34];
RZZ(0.5*pi) q[8],q[24];
RZZ(0.5*pi) q[18],q[9];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[35],q[14];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[19],q[39];
RZZ(0.5*pi) q[21],q[36];
RZZ(0.5*pi) q[37],q[26];
RZZ(0.5*pi) q[27],q[38];
RZZ(0.5*pi) q[29],q[28];
RZZ(0.5*pi) q[30],q[31];
U1q(0.583837953628663*pi,0.8806329573686069*pi) q[0];
U1q(0.737149147888222*pi,0.17781562866154*pi) q[1];
U1q(0.652817268524082*pi,1.49616124491362*pi) q[2];
U1q(0.683004228911433*pi,1.9860937713057831*pi) q[3];
U1q(0.456267695759508*pi,0.4532422437225696*pi) q[4];
U1q(0.188722290752147*pi,0.3980873020499196*pi) q[5];
U1q(0.756896276674664*pi,1.3100779132489002*pi) q[6];
U1q(0.863965230568548*pi,0.8217783963887602*pi) q[7];
U1q(0.609588421697173*pi,1.7823510800928108*pi) q[8];
U1q(0.58379758712402*pi,0.6200294444588677*pi) q[9];
U1q(0.487188656083682*pi,1.8214763714142297*pi) q[10];
U1q(0.572080138947011*pi,1.9988547312358094*pi) q[11];
U1q(0.263623065706007*pi,1.2510538988186184*pi) q[12];
U1q(0.60012513196666*pi,1.3946199449677916*pi) q[13];
U1q(0.306864217971461*pi,1.9000029876750109*pi) q[14];
U1q(0.609258441540773*pi,0.002190039940879629*pi) q[15];
U1q(0.315140105137922*pi,0.7354461662611396*pi) q[16];
U1q(0.410974453941399*pi,1.6619583843316974*pi) q[17];
U1q(0.0834721792914726*pi,0.5957320841231883*pi) q[18];
U1q(0.497129548787955*pi,1.71949704927221*pi) q[19];
U1q(0.37747638399983*pi,0.20773487277095004*pi) q[20];
U1q(0.745144036945909*pi,0.9042840856077001*pi) q[21];
U1q(0.641225593706339*pi,0.31259384252958977*pi) q[22];
U1q(0.569123984063865*pi,0.19565122291423087*pi) q[23];
U1q(0.409393186317937*pi,0.6748816718710264*pi) q[24];
U1q(0.46658040821224*pi,1.6816998536705476*pi) q[25];
U1q(0.785699622143975*pi,1.792317212788082*pi) q[26];
U1q(0.904247065352531*pi,1.6451466574431244*pi) q[27];
U1q(0.878697395782377*pi,0.4260925020952735*pi) q[28];
U1q(0.240636943792338*pi,0.5894105009917698*pi) q[29];
U1q(0.402672980023896*pi,1.8733191431841796*pi) q[30];
U1q(0.576414023391659*pi,1.2136822183006029*pi) q[31];
U1q(0.900431416625789*pi,1.3403844014564799*pi) q[32];
U1q(0.604982034767628*pi,0.3116887630028975*pi) q[33];
U1q(0.359185575283296*pi,0.28828395686030994*pi) q[34];
U1q(0.628436217073749*pi,1.5695130687293997*pi) q[35];
U1q(0.369489962165831*pi,0.9397700619517888*pi) q[36];
U1q(0.0204401772934044*pi,1.149239976814039*pi) q[37];
U1q(0.365688966488489*pi,1.8881060034988728*pi) q[38];
U1q(0.362378626155503*pi,0.8375496077554994*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[35],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[9],q[3];
RZZ(0.5*pi) q[4],q[17];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[32],q[6];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[10],q[24];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[28],q[13];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[34],q[16];
RZZ(0.5*pi) q[18],q[38];
RZZ(0.5*pi) q[37],q[19];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[29],q[23];
RZZ(0.5*pi) q[39],q[33];
U1q(0.660732829336989*pi,0.5687731009600654*pi) q[0];
U1q(0.555489162812976*pi,1.3504551326973804*pi) q[1];
U1q(0.451577960752675*pi,0.051260847457079706*pi) q[2];
U1q(0.433249348940793*pi,1.9877157391988831*pi) q[3];
U1q(0.112601495145598*pi,1.9122266164888*pi) q[4];
U1q(0.27297993824733*pi,0.9523214055706699*pi) q[5];
U1q(0.564446261716904*pi,1.8268566715084997*pi) q[6];
U1q(0.689086606410609*pi,1.4657760083994997*pi) q[7];
U1q(0.487951688333253*pi,0.31733050069152924*pi) q[8];
U1q(0.348059080779468*pi,1.264936099772898*pi) q[9];
U1q(0.578356537495614*pi,0.8891618042973004*pi) q[10];
U1q(0.664829593072408*pi,1.3794483039728291*pi) q[11];
U1q(0.288134010402032*pi,1.5962986854375085*pi) q[12];
U1q(0.448340586370473*pi,1.0113171073284324*pi) q[13];
U1q(0.295408391988184*pi,1.4180344358256907*pi) q[14];
U1q(0.510865601319927*pi,0.01464410978664965*pi) q[15];
U1q(0.417330263143056*pi,0.9084778810187295*pi) q[16];
U1q(0.511986564768255*pi,0.6364824640176465*pi) q[17];
U1q(0.396112683177619*pi,0.06332822506065838*pi) q[18];
U1q(0.572156103900873*pi,0.26618799847173946*pi) q[19];
U1q(0.715657002904011*pi,1.0977658390319904*pi) q[20];
U1q(0.590697521653257*pi,0.6339764593152193*pi) q[21];
U1q(0.343089922695312*pi,1.1822393746166995*pi) q[22];
U1q(0.979866198179651*pi,1.420060065951331*pi) q[23];
U1q(0.461182637902319*pi,0.13892851324188626*pi) q[24];
U1q(0.421446205362603*pi,1.7711169582278972*pi) q[25];
U1q(0.469145969815136*pi,0.5395623839159427*pi) q[26];
U1q(0.156730442847759*pi,1.6997578178151542*pi) q[27];
U1q(0.165911661861428*pi,1.3653192352160337*pi) q[28];
U1q(0.573059916463288*pi,1.8271498828368102*pi) q[29];
U1q(0.592084093271942*pi,1.4882955106533604*pi) q[30];
U1q(0.422658118949716*pi,1.6180875931053027*pi) q[31];
U1q(0.552192484792362*pi,1.46349070574463*pi) q[32];
U1q(0.307184649837763*pi,0.6002423066095472*pi) q[33];
U1q(0.450447599249643*pi,0.5319263021942007*pi) q[34];
U1q(0.499267235136186*pi,0.8558062609395396*pi) q[35];
U1q(0.401434660411661*pi,1.7361487674362497*pi) q[36];
U1q(0.809117562694439*pi,1.6406887928484384*pi) q[37];
U1q(0.202899961966801*pi,1.2635045444470023*pi) q[38];
U1q(0.653504923930285*pi,1.3426824137806896*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[1],q[33];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[21];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[34],q[5];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[35],q[10];
RZZ(0.5*pi) q[12],q[11];
RZZ(0.5*pi) q[19],q[13];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[16],q[27];
RZZ(0.5*pi) q[36],q[17];
RZZ(0.5*pi) q[28],q[20];
RZZ(0.5*pi) q[26],q[22];
RZZ(0.5*pi) q[25],q[38];
RZZ(0.5*pi) q[32],q[29];
U1q(0.525153879074882*pi,1.2212324893786857*pi) q[0];
U1q(0.501517897919848*pi,0.9151990669540595*pi) q[1];
U1q(0.692213480934672*pi,1.2621395759652998*pi) q[2];
U1q(0.448715342412913*pi,0.9838635031078429*pi) q[3];
U1q(0.554519722608834*pi,0.38924739820300047*pi) q[4];
U1q(0.245995274378845*pi,0.39594047759059947*pi) q[5];
U1q(0.17533781983647*pi,0.6678822277080005*pi) q[6];
U1q(0.583433140446647*pi,0.013179814745599927*pi) q[7];
U1q(0.540441652011967*pi,0.37402986789540016*pi) q[8];
U1q(0.51128362599504*pi,0.568076119611149*pi) q[9];
U1q(0.633165239685121*pi,1.9041202849933008*pi) q[10];
U1q(0.637213213472442*pi,0.5011230006977687*pi) q[11];
U1q(0.362142368717048*pi,0.5914774756422094*pi) q[12];
U1q(0.378014659607301*pi,1.866690302943132*pi) q[13];
U1q(0.484330004914861*pi,1.8358910772454493*pi) q[14];
U1q(0.207025230463228*pi,0.7616851698254692*pi) q[15];
U1q(0.408724025202897*pi,1.3834364360617997*pi) q[16];
U1q(0.554318975148987*pi,0.6804866803315477*pi) q[17];
U1q(0.266105170137506*pi,1.1636268047298088*pi) q[18];
U1q(0.35076380937757*pi,0.5067902313687007*pi) q[19];
U1q(0.434193012349579*pi,1.19968171185059*pi) q[20];
U1q(0.397386985998724*pi,1.6037613012658998*pi) q[21];
U1q(0.407401112716117*pi,0.24593902238829912*pi) q[22];
U1q(0.56201609951211*pi,1.282462755194711*pi) q[23];
U1q(0.695730220476925*pi,0.00761438064608555*pi) q[24];
U1q(0.64257521192735*pi,1.892163788515468*pi) q[25];
U1q(0.44271635400337*pi,1.6484159936054112*pi) q[26];
U1q(0.547894896918575*pi,1.0518319223062047*pi) q[27];
U1q(0.346170956768126*pi,0.4905022013438938*pi) q[28];
U1q(0.513501874922488*pi,0.7489416392981205*pi) q[29];
U1q(0.544914789896312*pi,0.0434422102263996*pi) q[30];
U1q(0.861686667328185*pi,0.40913475852728354*pi) q[31];
U1q(0.390798556521786*pi,0.10298601123770013*pi) q[32];
U1q(0.768510242530154*pi,1.0875856790015366*pi) q[33];
U1q(0.261354614791235*pi,0.5131285206644005*pi) q[34];
U1q(0.242669268184285*pi,1.1200335301718098*pi) q[35];
U1q(0.396672981156917*pi,0.12650595627401984*pi) q[36];
U1q(0.311427929031635*pi,1.5748334970870488*pi) q[37];
U1q(0.53377732711154*pi,0.6172215220568127*pi) q[38];
U1q(0.508084229386149*pi,0.23921838399127893*pi) q[39];
RZZ(0.5*pi) q[0],q[15];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[26],q[2];
RZZ(0.5*pi) q[22],q[3];
RZZ(0.5*pi) q[4],q[10];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[28],q[12];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[32],q[14];
RZZ(0.5*pi) q[39],q[16];
RZZ(0.5*pi) q[37],q[17];
RZZ(0.5*pi) q[25],q[20];
RZZ(0.5*pi) q[23],q[34];
RZZ(0.5*pi) q[27],q[24];
RZZ(0.5*pi) q[33],q[31];
U1q(0.181974805727528*pi,1.1933362336471856*pi) q[0];
U1q(0.127032076795163*pi,1.7265121962715*pi) q[1];
U1q(0.0757344232156926*pi,0.3262800321795005*pi) q[2];
U1q(0.373925104964709*pi,1.5610674664728528*pi) q[3];
U1q(0.423798214476512*pi,1.687522222808301*pi) q[4];
U1q(0.477917806975954*pi,1.0297934820390005*pi) q[5];
U1q(0.376147322868733*pi,1.0171380507748005*pi) q[6];
U1q(0.251044372188113*pi,1.9694297395547*pi) q[7];
U1q(0.71305376319276*pi,1.8202023952978*pi) q[8];
U1q(0.833649599929716*pi,1.101228329930148*pi) q[9];
U1q(0.430226568624241*pi,0.4934010076109985*pi) q[10];
U1q(0.502537532061524*pi,0.38988606578456597*pi) q[11];
U1q(0.109616928887489*pi,1.0924495596151083*pi) q[12];
U1q(0.964129438227757*pi,1.4160229953154317*pi) q[13];
U1q(0.45091175105987*pi,1.6215774410232502*pi) q[14];
U1q(0.558254996340693*pi,1.0106375433041705*pi) q[15];
U1q(0.312001403866606*pi,1.0373407255968008*pi) q[16];
U1q(0.348142572375628*pi,0.365854824413848*pi) q[17];
U1q(0.40678323841937*pi,1.6193167989394084*pi) q[18];
U1q(0.674998689427165*pi,1.9983011745203*pi) q[19];
U1q(0.76177600866189*pi,1.2506844274359796*pi) q[20];
U1q(0.434624789971381*pi,0.2968493705608992*pi) q[21];
U1q(0.345294206130819*pi,0.5658649350495004*pi) q[22];
U1q(0.483878555432255*pi,1.8003909856829114*pi) q[23];
U1q(0.565554383652866*pi,1.7410451180938864*pi) q[24];
U1q(0.768525281572178*pi,1.675933940730019*pi) q[25];
U1q(0.661453020084067*pi,1.303956178250612*pi) q[26];
U1q(0.545555188520754*pi,0.9201208404532046*pi) q[27];
U1q(0.44536579285192*pi,0.7985498652930136*pi) q[28];
U1q(0.301269199615039*pi,0.7329537324052993*pi) q[29];
U1q(0.303455196462921*pi,1.876384674954*pi) q[30];
U1q(0.19731968018287*pi,0.8748473604793823*pi) q[31];
U1q(0.263693265723886*pi,1.5640923662834005*pi) q[32];
U1q(0.292282946815446*pi,0.593180580598407*pi) q[33];
U1q(0.485746635884116*pi,0.16758933350570082*pi) q[34];
U1q(0.831893246681211*pi,1.2730563275778*pi) q[35];
U1q(0.57438912768354*pi,1.7538413587966204*pi) q[36];
U1q(0.464934445765259*pi,1.059575152414748*pi) q[37];
U1q(0.898923952984163*pi,1.2031726406680114*pi) q[38];
U1q(0.622587219109973*pi,1.3659885828329799*pi) q[39];
rz(1.8500422400505148*pi) q[0];
rz(2.7402349557224994*pi) q[1];
rz(0.9585422856935999*pi) q[2];
rz(3.7445959933301474*pi) q[3];
rz(3.9799596680829*pi) q[4];
rz(2.0381623195182*pi) q[5];
rz(0.045128831874901465*pi) q[6];
rz(1.9609318584723*pi) q[7];
rz(2.8751010177757*pi) q[8];
rz(1.7059520997007525*pi) q[9];
rz(2.0316786449983013*pi) q[10];
rz(1.3047035835772327*pi) q[11];
rz(3.830183088835291*pi) q[12];
rz(2.2865614096718687*pi) q[13];
rz(1.849568895137951*pi) q[14];
rz(0.5595760397285297*pi) q[15];
rz(3.7523634355683004*pi) q[16];
rz(3.165777260040853*pi) q[17];
rz(0.14940703967249291*pi) q[18];
rz(3.9308479137194006*pi) q[19];
rz(2.0794588520814*pi) q[20];
rz(3.7968765464898*pi) q[21];
rz(0.17048426671420103*pi) q[22];
rz(2.915307166427988*pi) q[23];
rz(0.11509125959661404*pi) q[24];
rz(2.5984709524442824*pi) q[25];
rz(3.468718327279488*pi) q[26];
rz(2.4708077670710953*pi) q[27];
rz(3.476704028797487*pi) q[28];
rz(0.1228624522725994*pi) q[29];
rz(0.41023588276570067*pi) q[30];
rz(2.962584920836118*pi) q[31];
rz(2.8129524826050005*pi) q[32];
rz(3.692378406715692*pi) q[33];
rz(0.6459890966985*pi) q[34];
rz(1.2008960417450005*pi) q[35];
rz(0.9533228663588815*pi) q[36];
rz(2.5559805308865506*pi) q[37];
rz(0.8405375910028887*pi) q[38];
rz(3.97867324198012*pi) q[39];
U1q(0.181974805727528*pi,0.0433784736977153*pi) q[0];
U1q(1.12703207679516*pi,1.466747151994039*pi) q[1];
U1q(1.07573442321569*pi,0.284822317873173*pi) q[2];
U1q(1.37392510496471*pi,0.305663459803001*pi) q[3];
U1q(0.423798214476512*pi,0.667481890891201*pi) q[4];
U1q(0.477917806975954*pi,0.0679558015571401*pi) q[5];
U1q(0.376147322868733*pi,0.0622668826497232*pi) q[6];
U1q(0.251044372188113*pi,0.9303615980269699*pi) q[7];
U1q(1.71305376319276*pi,1.69530341307347*pi) q[8];
U1q(1.83364959992972*pi,1.807180429630848*pi) q[9];
U1q(0.430226568624241*pi,1.525079652609292*pi) q[10];
U1q(0.502537532061524*pi,0.69458964936175*pi) q[11];
U1q(1.10961692888749*pi,1.9226326484504486*pi) q[12];
U1q(1.96412943822776*pi,0.702584404987266*pi) q[13];
U1q(0.45091175105987*pi,0.47114633616120005*pi) q[14];
U1q(1.55825499634069*pi,0.570213583032719*pi) q[15];
U1q(0.312001403866606*pi,1.789704161165138*pi) q[16];
U1q(1.34814257237563*pi,0.531632084454753*pi) q[17];
U1q(1.40678323841937*pi,0.7687238386119299*pi) q[18];
U1q(1.67499868942717*pi,0.929149088239781*pi) q[19];
U1q(1.76177600866189*pi,0.330143279517363*pi) q[20];
U1q(0.434624789971381*pi,1.09372591705073*pi) q[21];
U1q(1.34529420613082*pi,1.736349201763707*pi) q[22];
U1q(0.483878555432255*pi,1.7156981521109431*pi) q[23];
U1q(0.565554383652866*pi,0.856136377690524*pi) q[24];
U1q(0.768525281572178*pi,1.274404893174337*pi) q[25];
U1q(0.661453020084067*pi,1.772674505530164*pi) q[26];
U1q(0.545555188520754*pi,0.390928607524334*pi) q[27];
U1q(1.44536579285192*pi,1.275253894090481*pi) q[28];
U1q(0.301269199615039*pi,1.855816184677924*pi) q[29];
U1q(1.30345519646292*pi,1.286620557719649*pi) q[30];
U1q(0.19731968018287*pi,0.83743228131552*pi) q[31];
U1q(0.263693265723886*pi,1.377044848888396*pi) q[32];
U1q(1.29228294681545*pi,1.285558987314013*pi) q[33];
U1q(0.485746635884116*pi,1.813578430204275*pi) q[34];
U1q(0.831893246681211*pi,1.473952369322749*pi) q[35];
U1q(0.57438912768354*pi,1.7071642251555161*pi) q[36];
U1q(0.464934445765259*pi,0.61555568330131*pi) q[37];
U1q(1.89892395298416*pi,1.04371023167091*pi) q[38];
U1q(1.62258721910997*pi,0.344661824813049*pi) q[39];
RZZ(0.5*pi) q[0],q[15];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[26],q[2];
RZZ(0.5*pi) q[22],q[3];
RZZ(0.5*pi) q[4],q[10];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[28],q[12];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[32],q[14];
RZZ(0.5*pi) q[39],q[16];
RZZ(0.5*pi) q[37],q[17];
RZZ(0.5*pi) q[25],q[20];
RZZ(0.5*pi) q[23],q[34];
RZZ(0.5*pi) q[27],q[24];
RZZ(0.5*pi) q[33],q[31];
U1q(0.525153879074882*pi,0.0712747294291943*pi) q[0];
U1q(3.498482102080152*pi,1.278060281311481*pi) q[1];
U1q(3.3077865190653277*pi,0.34896277408739473*pi) q[2];
U1q(1.44871534241291*pi,1.8828674231680629*pi) q[3];
U1q(0.554519722608834*pi,0.36920706628585*pi) q[4];
U1q(0.245995274378845*pi,1.434102797108773*pi) q[5];
U1q(0.17533781983647*pi,1.71301105958289*pi) q[6];
U1q(0.583433140446647*pi,1.9741116732178399*pi) q[7];
U1q(3.459558347988033*pi,0.14147594047583978*pi) q[8];
U1q(3.48871637400496*pi,0.3403326399498199*pi) q[9];
U1q(1.63316523968512*pi,1.9357989299915999*pi) q[10];
U1q(0.637213213472442*pi,0.8058265842749299*pi) q[11];
U1q(3.637857631282952*pi,0.42360473242334507*pi) q[12];
U1q(1.3780146596073*pi,0.25191709735956724*pi) q[13];
U1q(1.48433000491486*pi,1.68545997238341*pi) q[14];
U1q(1.20702523046323*pi,0.8191659565113536*pi) q[15];
U1q(0.408724025202897*pi,0.1357998716301101*pi) q[16];
U1q(3.554318975148986*pi,0.21700022853708023*pi) q[17];
U1q(3.733894829862494*pi,0.22441383282150662*pi) q[18];
U1q(3.64923619062243*pi,1.420660031391392*pi) q[19];
U1q(3.565806987650421*pi,1.3811459951027487*pi) q[20];
U1q(1.39738698599872*pi,0.400637847755654*pi) q[21];
U1q(1.40740111271612*pi,1.0562751144249083*pi) q[22];
U1q(1.56201609951211*pi,0.197769921622773*pi) q[23];
U1q(1.69573022047693*pi,0.12270564024268005*pi) q[24];
U1q(1.64257521192735*pi,1.490634740959781*pi) q[25];
U1q(1.44271635400337*pi,0.11713432088497*pi) q[26];
U1q(0.547894896918575*pi,1.522639689377318*pi) q[27];
U1q(1.34617095676813*pi,0.5833015580395708*pi) q[28];
U1q(1.51350187492249*pi,1.8718040915707501*pi) q[29];
U1q(3.4550852101036877*pi,1.1195630224472188*pi) q[30];
U1q(1.86168666732819*pi,0.371719679363464*pi) q[31];
U1q(0.390798556521786*pi,0.91593849384268*pi) q[32];
U1q(3.231489757469846*pi,1.79115388891084*pi) q[33];
U1q(3.261354614791235*pi,1.15911761736298*pi) q[34];
U1q(1.24266926818429*pi,1.32092957191677*pi) q[35];
U1q(1.39667298115692*pi,1.0798288226328898*pi) q[36];
U1q(1.31142792903164*pi,1.1308140279735501*pi) q[37];
U1q(3.4662226728884598*pi,0.6296613502820466*pi) q[38];
U1q(1.50808422938615*pi,1.4714320236547493*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[1],q[33];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[21];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[34],q[5];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[35],q[10];
RZZ(0.5*pi) q[12],q[11];
RZZ(0.5*pi) q[19],q[13];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[16],q[27];
RZZ(0.5*pi) q[36],q[17];
RZZ(0.5*pi) q[28],q[20];
RZZ(0.5*pi) q[26],q[22];
RZZ(0.5*pi) q[25],q[38];
RZZ(0.5*pi) q[32],q[29];
U1q(1.66073282933699*pi,1.41881534101057*pi) q[0];
U1q(3.444510837187024*pi,1.842804215568167*pi) q[1];
U1q(3.548422039247325*pi,0.5598415025956252*pi) q[2];
U1q(0.433249348940793*pi,0.8867196592591067*pi) q[3];
U1q(1.1126014951456*pi,1.89218628457163*pi) q[4];
U1q(1.27297993824733*pi,1.9904837250888199*pi) q[5];
U1q(0.564446261716904*pi,0.8719855033833501*pi) q[6];
U1q(0.689086606410609*pi,1.4267078668717899*pi) q[7];
U1q(1.48795168833325*pi,1.1981753076797201*pi) q[8];
U1q(1.34805908077947*pi,1.6434726597880633*pi) q[9];
U1q(1.57835653749561*pi,1.9507574106876442*pi) q[10];
U1q(0.664829593072408*pi,1.6841518875500103*pi) q[11];
U1q(3.288134010402032*pi,0.4187835226281166*pi) q[12];
U1q(3.448340586370473*pi,1.3965439017448613*pi) q[13];
U1q(3.7045916080118158*pi,1.103316613803207*pi) q[14];
U1q(0.510865601319927*pi,0.0721248964724886*pi) q[15];
U1q(1.41733026314306*pi,0.6608413165870402*pi) q[16];
U1q(1.51198656476826*pi,0.17299601222319327*pi) q[17];
U1q(3.603887316822381*pi,0.32471241249065974*pi) q[18];
U1q(3.427843896099127*pi,1.661262264288391*pi) q[19];
U1q(3.284342997095989*pi,1.4830618679213545*pi) q[20];
U1q(3.409302478346743*pi,0.3704226897062881*pi) q[21];
U1q(0.343089922695312*pi,1.9925754666533382*pi) q[22];
U1q(3.020133801820351*pi,0.06017261086619552*pi) q[23];
U1q(3.538817362097682*pi,0.9913915076468887*pi) q[24];
U1q(1.4214462053626*pi,0.6116815712473502*pi) q[25];
U1q(3.530854030184864*pi,0.2259879305744681*pi) q[26];
U1q(1.15673044284776*pi,1.170565584886282*pi) q[27];
U1q(3.1659116618614283*pi,1.458118591911711*pi) q[28];
U1q(3.426940083536712*pi,1.793595848032063*pi) q[29];
U1q(1.59208409327194*pi,0.6747097220202667*pi) q[30];
U1q(3.422658118949716*pi,1.1627668447854593*pi) q[31];
U1q(1.55219248479236*pi,0.2764431883496099*pi) q[32];
U1q(3.692815350162237*pi,1.2784972613028298*pi) q[33];
U1q(3.549552400750357*pi,0.14031983583326135*pi) q[34];
U1q(1.49926723513619*pi,0.5851568411490371*pi) q[35];
U1q(1.40143466041166*pi,1.470186011470675*pi) q[36];
U1q(3.190882437305561*pi,0.06495873221212989*pi) q[37];
U1q(3.202899961966801*pi,1.983378327891883*pi) q[38];
U1q(3.653504923930284*pi,0.5748960534441907*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[35],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[9],q[3];
RZZ(0.5*pi) q[4],q[17];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[32],q[6];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[10],q[24];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[28],q[13];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[34],q[16];
RZZ(0.5*pi) q[18],q[38];
RZZ(0.5*pi) q[37],q[19];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[29],q[23];
RZZ(0.5*pi) q[39],q[33];
U1q(3.416162046371337*pi,0.10695548460203486*pi) q[0];
U1q(1.73714914788822*pi,0.015443719604012252*pi) q[1];
U1q(3.347182731475918*pi,1.1149411051390903*pi) q[2];
U1q(1.68300422891143*pi,1.8850976913660058*pi) q[3];
U1q(3.543732304240492*pi,1.351170657337835*pi) q[4];
U1q(3.811277709247853*pi,1.5447178286095697*pi) q[5];
U1q(0.756896276674664*pi,0.3552067451237799*pi) q[6];
U1q(0.863965230568548*pi,0.7827102548610299*pi) q[7];
U1q(0.609588421697173*pi,0.6631958870809962*pi) q[8];
U1q(0.58379758712402*pi,1.9985660044740428*pi) q[9];
U1q(0.487188656083682*pi,0.8830719778046037*pi) q[10];
U1q(1.57208013894701*pi,0.3035583148129901*pi) q[11];
U1q(0.263623065706007*pi,0.07353873600926963*pi) q[12];
U1q(3.39987486803334*pi,1.013241064105462*pi) q[13];
U1q(3.6931357820285378*pi,1.6213480619538867*pi) q[14];
U1q(0.609258441540773*pi,1.0596708266267245*pi) q[15];
U1q(1.31514010513792*pi,1.8338730313446305*pi) q[16];
U1q(1.4109744539414*pi,0.14752009190918092*pi) q[17];
U1q(3.083472179291472*pi,0.7923085534281273*pi) q[18];
U1q(3.502870451212045*pi,1.2079532134879212*pi) q[19];
U1q(3.37747638399983*pi,1.373092834182394*pi) q[20];
U1q(1.74514403694591*pi,1.1001150634138182*pi) q[21];
U1q(0.641225593706339*pi,0.12292993456620804*pi) q[22];
U1q(1.56912398406386*pi,1.2845814539032911*pi) q[23];
U1q(1.40939318631794*pi,1.4554383490177107*pi) q[24];
U1q(1.46658040821224*pi,1.52226446668999*pi) q[25];
U1q(1.78569962214398*pi,1.973233101702327*pi) q[26];
U1q(1.90424706535253*pi,0.2251767452583111*pi) q[27];
U1q(1.87869739578238*pi,1.3973453250324628*pi) q[28];
U1q(3.759363056207662*pi,0.031335229877106185*pi) q[29];
U1q(3.4026729800238957*pi,0.05973335455109363*pi) q[30];
U1q(1.57641402339166*pi,0.7583614699807655*pi) q[31];
U1q(3.9004314166257896*pi,0.3995494926377565*pi) q[32];
U1q(3.395017965232372*pi,0.5670508049094758*pi) q[33];
U1q(1.3591855752833*pi,0.3839621811671141*pi) q[34];
U1q(0.628436217073749*pi,1.2988636489388972*pi) q[35];
U1q(0.369489962165831*pi,0.6738073059862151*pi) q[36];
U1q(3.979559822706596*pi,1.55640754824653*pi) q[37];
U1q(0.365688966488489*pi,0.6079797869437551*pi) q[38];
U1q(3.637621373844496*pi,1.080028859469384*pi) q[39];
RZZ(0.5*pi) q[0],q[16];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[4],q[7];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[6],q[34];
RZZ(0.5*pi) q[8],q[24];
RZZ(0.5*pi) q[18],q[9];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[35],q[14];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[19],q[39];
RZZ(0.5*pi) q[21],q[36];
RZZ(0.5*pi) q[37],q[26];
RZZ(0.5*pi) q[27],q[38];
RZZ(0.5*pi) q[29],q[28];
RZZ(0.5*pi) q[30],q[31];
U1q(3.592315792831457*pi,1.0511034059810547*pi) q[0];
U1q(0.777402668942508*pi,0.7454155302634922*pi) q[1];
U1q(3.543647896557245*pi,1.7889697988900353*pi) q[2];
U1q(3.586837004093232*pi,0.9047002783104232*pi) q[3];
U1q(3.441073317836361*pi,1.5471821400455161*pi) q[4];
U1q(3.8652364317677392*pi,1.7074670784200476*pi) q[5];
U1q(1.58896199190066*pi,1.32376837880817*pi) q[6];
U1q(0.630274028447919*pi,1.3556413055538599*pi) q[7];
U1q(0.512182444603415*pi,0.2758717750759989*pi) q[8];
U1q(0.248142965611392*pi,1.5447601337892536*pi) q[9];
U1q(1.71401742281898*pi,0.6425984914600242*pi) q[10];
U1q(3.505532647731449*pi,1.3030399963010808*pi) q[11];
U1q(1.90278865104612*pi,0.7879568891077695*pi) q[12];
U1q(1.56812694209372*pi,0.46109242966272346*pi) q[13];
U1q(1.23290853557578*pi,0.37328549301257863*pi) q[14];
U1q(1.35794714450726*pi,0.6224610120468945*pi) q[15];
U1q(1.59528050656663*pi,0.5410556573739207*pi) q[16];
U1q(1.23499447234137*pi,0.3482579651871849*pi) q[17];
U1q(0.652621610214298*pi,0.24579517418356023*pi) q[18];
U1q(3.607982830358835*pi,1.2333686040750167*pi) q[19];
U1q(0.54787794365631*pi,0.16895874068362698*pi) q[20];
U1q(1.43291495723573*pi,1.355707728869457*pi) q[21];
U1q(1.42422385542818*pi,1.181792695031008*pi) q[22];
U1q(0.607456765577331*pi,0.5826443554098208*pi) q[23];
U1q(0.64230619945232*pi,0.45664807296273136*pi) q[24];
U1q(3.701409519547643*pi,1.2159821463066436*pi) q[25];
U1q(1.88979140677068*pi,1.7196315740164367*pi) q[26];
U1q(1.69918602041143*pi,0.4294927389881713*pi) q[27];
U1q(0.88571966855043*pi,1.6274671024502627*pi) q[28];
U1q(1.9378785918204*pi,1.461629878673117*pi) q[29];
U1q(3.718675865647211*pi,1.4254246458992648*pi) q[30];
U1q(1.58662498548511*pi,0.9116036532648129*pi) q[31];
U1q(1.807941432996*pi,0.7055791178913866*pi) q[32];
U1q(1.77582284054221*pi,1.9469211471987902*pi) q[33];
U1q(0.65910886547728*pi,0.4596317110118444*pi) q[34];
U1q(0.748192807185111*pi,1.269504300290587*pi) q[35];
U1q(0.946164591976671*pi,0.8108610936728855*pi) q[36];
U1q(3.302404815007045*pi,1.4711007986652387*pi) q[37];
U1q(1.85619464875402*pi,0.382179282778333*pi) q[38];
U1q(3.474903362715228*pi,1.3310276257724238*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[7],q[25];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[13],q[12];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[38],q[17];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[31],q[24];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[34],q[28];
RZZ(0.5*pi) q[32],q[37];
U1q(3.626879503468211*pi,1.5859098994440046*pi) q[0];
U1q(0.246031181953476*pi,1.4823050126636321*pi) q[1];
U1q(1.77445116959928*pi,1.3812595286275327*pi) q[2];
U1q(0.449941815843335*pi,1.0583287232265532*pi) q[3];
U1q(3.363151151987262*pi,0.5579936674694865*pi) q[4];
U1q(1.46046654083238*pi,0.31013783236976744*pi) q[5];
U1q(3.1332947674773513*pi,1.6067235867557237*pi) q[6];
U1q(1.88946919540139*pi,0.37516248971419053*pi) q[7];
U1q(1.27990683455447*pi,1.1036678291618696*pi) q[8];
U1q(0.170496295367881*pi,0.11269798593522395*pi) q[9];
U1q(1.57688090362493*pi,1.0221231614715274*pi) q[10];
U1q(3.59099178657476*pi,0.9611021881506412*pi) q[11];
U1q(1.62606721909024*pi,1.389679149329071*pi) q[12];
U1q(1.25224521024324*pi,1.1118674520416238*pi) q[13];
U1q(0.364637318501898*pi,0.2674107730373887*pi) q[14];
U1q(1.38232462571931*pi,0.253883538259448*pi) q[15];
U1q(3.174926089827517*pi,1.919831546962178*pi) q[16];
U1q(1.12189800635965*pi,0.25232434086375743*pi) q[17];
U1q(0.437059336714011*pi,0.6637409378757173*pi) q[18];
U1q(0.747075627737868*pi,1.8767219062658467*pi) q[19];
U1q(1.25668763349311*pi,0.1350638009090419*pi) q[20];
U1q(3.533134371419049*pi,0.7784801805412068*pi) q[21];
U1q(1.28655688345214*pi,0.7453759532850341*pi) q[22];
U1q(3.409043988065733*pi,0.5429957596058816*pi) q[23];
U1q(3.5888620922784282*pi,1.751106384310301*pi) q[24];
U1q(0.651432422593109*pi,1.8134129619650228*pi) q[25];
U1q(3.437756167758176*pi,0.07199433443564196*pi) q[26];
U1q(1.26529761801025*pi,1.8785244303175332*pi) q[27];
U1q(1.86644353794349*pi,1.817572004382943*pi) q[28];
U1q(1.42701851066514*pi,1.0800386198998*pi) q[29];
U1q(3.276512504709134*pi,1.6558665389213938*pi) q[30];
U1q(0.534445586322182*pi,0.8403899314475427*pi) q[31];
U1q(1.32963637362217*pi,0.4345130572529843*pi) q[32];
U1q(3.330203750587457*pi,1.9117606588485305*pi) q[33];
U1q(1.42135030743185*pi,0.5996764408067339*pi) q[34];
U1q(0.262521031339458*pi,1.8136195048951578*pi) q[35];
U1q(1.13820547290904*pi,1.755652853135917*pi) q[36];
U1q(0.472805077145162*pi,0.5630605115432883*pi) q[37];
U1q(3.1283589746413742*pi,1.977852109363896*pi) q[38];
U1q(1.52501138627456*pi,1.4159065906199686*pi) q[39];
RZZ(0.5*pi) q[0],q[13];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[23];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[16],q[10];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[12],q[24];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[37],q[15];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[21],q[31];
RZZ(0.5*pi) q[30],q[29];
RZZ(0.5*pi) q[32],q[38];
RZZ(0.5*pi) q[39],q[36];
U1q(1.22607242943562*pi,1.7140639152443518*pi) q[0];
U1q(0.504537522341449*pi,1.8236460873582327*pi) q[1];
U1q(0.457217201088832*pi,0.4488698319938731*pi) q[2];
U1q(0.735971503853859*pi,0.2782654574418135*pi) q[3];
U1q(0.177978091942841*pi,1.5627097371986558*pi) q[4];
U1q(1.38190303588882*pi,0.5690416140171255*pi) q[5];
U1q(3.285077272781546*pi,0.5137367838771949*pi) q[6];
U1q(1.26108342148392*pi,0.21708055939788906*pi) q[7];
U1q(1.45140881673983*pi,0.2765278568477658*pi) q[8];
U1q(0.453916177281697*pi,0.18883541237632429*pi) q[9];
U1q(0.39295029666598*pi,1.2209738551640363*pi) q[10];
U1q(1.67851357637008*pi,1.1654905443557269*pi) q[11];
U1q(0.754423884104489*pi,0.7465821559941612*pi) q[12];
U1q(1.60895463426798*pi,0.014425176734319756*pi) q[13];
U1q(0.798182321915926*pi,0.49477345930390815*pi) q[14];
U1q(0.112227896741101*pi,1.670580465629218*pi) q[15];
U1q(3.725200416262469*pi,1.3679081536388074*pi) q[16];
U1q(0.278182464396445*pi,0.6931692302164776*pi) q[17];
U1q(0.629543550874243*pi,0.5697879999033493*pi) q[18];
U1q(0.402552435403241*pi,0.1374018158116166*pi) q[19];
U1q(1.70933989283863*pi,0.29516914571771435*pi) q[20];
U1q(0.208190733537805*pi,1.1590475930413078*pi) q[21];
U1q(0.780919435302561*pi,1.3876724731140841*pi) q[22];
U1q(1.81455597917983*pi,1.4832661097087891*pi) q[23];
U1q(1.70566619095162*pi,1.2319797883401336*pi) q[24];
U1q(0.724253696004692*pi,0.19736591758347277*pi) q[25];
U1q(1.72926018597708*pi,0.18506609616481562*pi) q[26];
U1q(0.482919520573873*pi,1.0245465696399227*pi) q[27];
U1q(1.15678998216674*pi,1.8130781481816856*pi) q[28];
U1q(1.21372069396063*pi,0.7954638383470254*pi) q[29];
U1q(1.46429500816529*pi,1.907752987614777*pi) q[30];
U1q(0.511910299908187*pi,0.22810855725313273*pi) q[31];
U1q(0.214978616314598*pi,0.17557321864589426*pi) q[32];
U1q(1.74386542705225*pi,0.23251330081874588*pi) q[33];
U1q(1.53686495004199*pi,1.8463372103600841*pi) q[34];
U1q(0.813901029743174*pi,0.5123543979828575*pi) q[35];
U1q(1.71561828524044*pi,0.10861448346438962*pi) q[36];
U1q(0.469189005290398*pi,1.145685915908948*pi) q[37];
U1q(1.62037023629113*pi,0.14475065694657063*pi) q[38];
U1q(0.20778960319836*pi,0.7986101482496295*pi) q[39];
rz(0.28593608475564825*pi) q[0];
rz(0.17635391264176725*pi) q[1];
rz(3.551130168006127*pi) q[2];
rz(3.7217345425581865*pi) q[3];
rz(0.4372902628013442*pi) q[4];
rz(3.4309583859828745*pi) q[5];
rz(3.486263216122805*pi) q[6];
rz(1.782919440602111*pi) q[7];
rz(1.7234721431522342*pi) q[8];
rz(1.8111645876236757*pi) q[9];
rz(2.7790261448359637*pi) q[10];
rz(0.8345094556442731*pi) q[11];
rz(3.2534178440058388*pi) q[12];
rz(1.9855748232656802*pi) q[13];
rz(1.5052265406960919*pi) q[14];
rz(2.329419534370782*pi) q[15];
rz(2.6320918463611926*pi) q[16];
rz(3.3068307697835224*pi) q[17];
rz(1.4302120000966507*pi) q[18];
rz(3.8625981841883834*pi) q[19];
rz(3.7048308542822856*pi) q[20];
rz(0.8409524069586922*pi) q[21];
rz(0.6123275268859159*pi) q[22];
rz(0.5167338902912109*pi) q[23];
rz(0.7680202116598664*pi) q[24];
rz(3.8026340824165272*pi) q[25];
rz(1.8149339038351844*pi) q[26];
rz(0.9754534303600773*pi) q[27];
rz(2.1869218518183144*pi) q[28];
rz(3.2045361616529746*pi) q[29];
rz(2.092247012385223*pi) q[30];
rz(3.7718914427468673*pi) q[31];
rz(1.8244267813541057*pi) q[32];
rz(3.767486699181254*pi) q[33];
rz(2.153662789639916*pi) q[34];
rz(1.4876456020171425*pi) q[35];
rz(3.8913855165356104*pi) q[36];
rz(0.8543140840910519*pi) q[37];
rz(3.8552493430534294*pi) q[38];
rz(1.2013898517503705*pi) q[39];
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