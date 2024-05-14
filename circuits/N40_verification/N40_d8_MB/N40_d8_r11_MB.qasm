OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(1.56284286396696*pi,0.7824851278579864*pi) q[0];
U1q(1.57609317509132*pi,0.48520343144104955*pi) q[1];
U1q(0.41038047969844*pi,0.304941339098896*pi) q[2];
U1q(0.4048544606538*pi,0.941675756691621*pi) q[3];
U1q(3.135186431571781*pi,1.4735289724861218*pi) q[4];
U1q(0.564520122788205*pi,0.344075272389743*pi) q[5];
U1q(0.505511683040376*pi,0.6940005963324001*pi) q[6];
U1q(0.538150167481349*pi,0.439887795767421*pi) q[7];
U1q(0.827984178846658*pi,1.9979024526990519*pi) q[8];
U1q(3.616289541781675*pi,0.8694901911534202*pi) q[9];
U1q(1.29148794209268*pi,0.4606516059878247*pi) q[10];
U1q(1.65005673084058*pi,1.581975174836118*pi) q[11];
U1q(0.581180661022703*pi,0.538832048204542*pi) q[12];
U1q(0.570558506627599*pi,1.255702442132326*pi) q[13];
U1q(1.442820335371*pi,0.07162468358554831*pi) q[14];
U1q(1.4973136805354*pi,1.669523385162377*pi) q[15];
U1q(1.17670536162564*pi,1.514021239680973*pi) q[16];
U1q(0.828302697900893*pi,0.945439299820098*pi) q[17];
U1q(0.199022540309589*pi,0.420609783090644*pi) q[18];
U1q(1.27538873249152*pi,1.1318237883683722*pi) q[19];
U1q(3.316728479737945*pi,1.0587952924105197*pi) q[20];
U1q(0.427922629936015*pi,1.821899740040278*pi) q[21];
U1q(1.52532303169377*pi,1.9120999631807007*pi) q[22];
U1q(0.471081487205849*pi,0.400499732748936*pi) q[23];
U1q(0.716164160597342*pi,0.300282085140319*pi) q[24];
U1q(1.58490134291172*pi,1.3057055251748833*pi) q[25];
U1q(3.443913735629144*pi,1.5621646890693668*pi) q[26];
U1q(1.13142028352966*pi,1.4777608067361727*pi) q[27];
U1q(0.978121690260516*pi,1.47799609443476*pi) q[28];
U1q(0.13911698688851*pi,0.330681734289884*pi) q[29];
U1q(1.53539002543189*pi,1.844790672095009*pi) q[30];
U1q(0.244028833813108*pi,0.561974522362515*pi) q[31];
U1q(0.350479382597176*pi,0.64592626728996*pi) q[32];
U1q(1.16742105568322*pi,0.6971034754820052*pi) q[33];
U1q(0.759650530034869*pi,0.232833711262544*pi) q[34];
U1q(1.82678978216519*pi,1.5226235742042311*pi) q[35];
U1q(1.56512550361914*pi,1.1226092230465752*pi) q[36];
U1q(1.18342030877322*pi,0.8709675885758634*pi) q[37];
U1q(3.617702877017847*pi,1.0371974682626985*pi) q[38];
U1q(0.777772534920656*pi,1.500282315355129*pi) q[39];
RZZ(0.5*pi) q[0],q[22];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[38],q[2];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[31],q[6];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[21],q[8];
RZZ(0.5*pi) q[39],q[9];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[15],q[12];
RZZ(0.5*pi) q[27],q[13];
RZZ(0.5*pi) q[35],q[14];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[29],q[18];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[32],q[24];
RZZ(0.5*pi) q[28],q[33];
U1q(0.688682680372143*pi,0.5134918369858061*pi) q[0];
U1q(0.0881974856227273*pi,1.4132523027220496*pi) q[1];
U1q(0.125763143972534*pi,1.92607640982015*pi) q[2];
U1q(0.334714061220401*pi,1.46311816913731*pi) q[3];
U1q(0.0833390163307269*pi,0.4675193402540816*pi) q[4];
U1q(0.393086060851122*pi,0.5991053286114099*pi) q[5];
U1q(0.798544497301137*pi,1.5644694664221999*pi) q[6];
U1q(0.393227467191231*pi,1.75463553089056*pi) q[7];
U1q(0.732478386389738*pi,0.2020536825016599*pi) q[8];
U1q(0.46047904698666*pi,1.7082500829626004*pi) q[9];
U1q(0.0760421391417979*pi,1.2516180939318549*pi) q[10];
U1q(0.171051025628983*pi,0.8247659041511981*pi) q[11];
U1q(0.563138941596323*pi,1.9602055796237998*pi) q[12];
U1q(0.632982848689047*pi,0.004925785211240097*pi) q[13];
U1q(0.345900493831726*pi,0.8382529228408182*pi) q[14];
U1q(0.476975744835322*pi,0.9175640892180468*pi) q[15];
U1q(0.102645335232739*pi,0.04948971862917295*pi) q[16];
U1q(0.31952937379813*pi,0.64204147662143*pi) q[17];
U1q(0.684888732385071*pi,0.47803065412505985*pi) q[18];
U1q(0.137440424141377*pi,1.2777558663647919*pi) q[19];
U1q(0.598889978077403*pi,1.47532923000501*pi) q[20];
U1q(0.344511610372858*pi,0.6733787024932099*pi) q[21];
U1q(0.183330612665077*pi,1.0035547386008905*pi) q[22];
U1q(0.233950065102103*pi,1.0994229618898461*pi) q[23];
U1q(0.248376433030504*pi,1.151179308506773*pi) q[24];
U1q(0.419872532915947*pi,1.2872475763111035*pi) q[25];
U1q(0.446518792182381*pi,0.5015742454662968*pi) q[26];
U1q(0.344274136954123*pi,0.4224519397444624*pi) q[27];
U1q(0.350135847036149*pi,1.729083607323901*pi) q[28];
U1q(0.471858953784483*pi,1.8913485020511*pi) q[29];
U1q(0.675764800931402*pi,1.635982235661309*pi) q[30];
U1q(0.366204017754761*pi,0.33780126203957006*pi) q[31];
U1q(0.771654229808907*pi,1.9675260278219637*pi) q[32];
U1q(0.441761886226983*pi,0.19028228799847513*pi) q[33];
U1q(0.310950738936571*pi,1.557039742604212*pi) q[34];
U1q(0.161030576509607*pi,0.23497067059369137*pi) q[35];
U1q(0.506489394520897*pi,1.510609430424695*pi) q[36];
U1q(0.236449754437332*pi,1.8734573919462836*pi) q[37];
U1q(0.341064418531402*pi,0.6509229122562985*pi) q[38];
U1q(0.550656299557688*pi,1.5557621686100296*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[1],q[26];
RZZ(0.5*pi) q[2],q[33];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[30],q[4];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[7],q[38];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[11],q[34];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[20],q[13];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[19],q[17];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[23],q[32];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[27],q[37];
U1q(0.912986474939097*pi,1.7995016277991063*pi) q[0];
U1q(0.134907140283775*pi,0.9876795535130096*pi) q[1];
U1q(0.644335483916534*pi,0.1831829003644998*pi) q[2];
U1q(0.505204730918356*pi,0.2501325070471898*pi) q[3];
U1q(0.471440909639951*pi,1.7138970602674517*pi) q[4];
U1q(0.64849842407387*pi,1.4805383279640303*pi) q[5];
U1q(0.370472455355003*pi,0.73591950892517*pi) q[6];
U1q(0.583392447665085*pi,1.06712446777634*pi) q[7];
U1q(0.53421893707065*pi,0.96261708134318*pi) q[8];
U1q(0.757096315837764*pi,1.25349715524316*pi) q[9];
U1q(0.641429545908852*pi,0.16164921453214465*pi) q[10];
U1q(0.645953099619999*pi,1.0401684598389078*pi) q[11];
U1q(0.246210158989552*pi,0.11409520129141981*pi) q[12];
U1q(0.777301027788081*pi,1.3810917614337699*pi) q[13];
U1q(0.275216315706205*pi,1.0109962445046787*pi) q[14];
U1q(0.419791462816206*pi,0.8078345768096078*pi) q[15];
U1q(0.903511354433318*pi,0.27212027124982363*pi) q[16];
U1q(0.331029854160943*pi,0.24345799804881985*pi) q[17];
U1q(0.535220234038726*pi,0.8967002897807301*pi) q[18];
U1q(0.438486946975339*pi,1.7287090039600121*pi) q[19];
U1q(0.852924935487905*pi,0.3854922594705199*pi) q[20];
U1q(0.783284560977616*pi,1.6421421517153298*pi) q[21];
U1q(0.571203351362003*pi,0.6268196353729412*pi) q[22];
U1q(0.372991852920928*pi,0.4690790892477201*pi) q[23];
U1q(0.403514187019004*pi,1.33463221017672*pi) q[24];
U1q(0.185292425593255*pi,0.5150010685767237*pi) q[25];
U1q(0.273335277145903*pi,1.6484094898023773*pi) q[26];
U1q(0.688777661146737*pi,0.43049245501110267*pi) q[27];
U1q(0.580621171643384*pi,0.22889486011901994*pi) q[28];
U1q(0.598533684074705*pi,0.23027047284034996*pi) q[29];
U1q(0.340815898311451*pi,1.9978211803027088*pi) q[30];
U1q(0.416063843115201*pi,0.6030338997863902*pi) q[31];
U1q(0.524871855690629*pi,0.10263517510287001*pi) q[32];
U1q(0.154659829026414*pi,1.638564896684835*pi) q[33];
U1q(0.71604978106556*pi,1.78399749481836*pi) q[34];
U1q(0.828293674635924*pi,1.1139179314894214*pi) q[35];
U1q(0.796207188275542*pi,1.0224557569800052*pi) q[36];
U1q(0.222846466608176*pi,0.47594310977261367*pi) q[37];
U1q(0.403414607307078*pi,0.7309665537948185*pi) q[38];
U1q(0.394732467798607*pi,1.8272598868248302*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[1],q[28];
RZZ(0.5*pi) q[2],q[8];
RZZ(0.5*pi) q[4],q[3];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[25],q[6];
RZZ(0.5*pi) q[36],q[7];
RZZ(0.5*pi) q[24],q[9];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[21],q[12];
RZZ(0.5*pi) q[30],q[14];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[20],q[17];
RZZ(0.5*pi) q[19],q[32];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[27],q[38];
RZZ(0.5*pi) q[29],q[33];
RZZ(0.5*pi) q[35],q[31];
U1q(0.0490003127194353*pi,0.9567013086498468*pi) q[0];
U1q(0.260003285106841*pi,0.9156097636617595*pi) q[1];
U1q(0.872732756754013*pi,0.23604461566914026*pi) q[2];
U1q(0.289037797195898*pi,1.7024234732049806*pi) q[3];
U1q(0.224752353470736*pi,0.05485237001827148*pi) q[4];
U1q(0.38359869131729*pi,1.49659770689874*pi) q[5];
U1q(0.227758760792784*pi,0.8892985189956901*pi) q[6];
U1q(0.217711532663194*pi,1.8853248543329695*pi) q[7];
U1q(0.366784109114019*pi,0.7585182379587394*pi) q[8];
U1q(0.492150927529565*pi,1.11552996294495*pi) q[9];
U1q(0.347499167575274*pi,1.242348116307685*pi) q[10];
U1q(0.520979512279954*pi,1.5280551652224972*pi) q[11];
U1q(0.726903531001162*pi,0.8647634574770899*pi) q[12];
U1q(0.439754166691238*pi,0.9692213675332706*pi) q[13];
U1q(0.617987929813689*pi,0.7732035630538778*pi) q[14];
U1q(0.435903848823576*pi,0.8324088544993575*pi) q[15];
U1q(0.29918442406572*pi,1.321727574577963*pi) q[16];
U1q(0.565614241133235*pi,1.5713277042505798*pi) q[17];
U1q(0.318676462577983*pi,1.5776219975381007*pi) q[18];
U1q(0.531150575756974*pi,1.7363539437659723*pi) q[19];
U1q(0.205965662457452*pi,1.1793016432320798*pi) q[20];
U1q(0.679753656471057*pi,0.6410392694842004*pi) q[21];
U1q(0.36999003704351*pi,1.3516946539124408*pi) q[22];
U1q(0.82090384228756*pi,1.9593556865384203*pi) q[23];
U1q(0.506663811699687*pi,1.73071828460417*pi) q[24];
U1q(0.766561760384119*pi,0.6844593112084736*pi) q[25];
U1q(0.74663110446396*pi,1.3081820970275668*pi) q[26];
U1q(0.352463395507483*pi,0.23509284172898326*pi) q[27];
U1q(0.80613565981111*pi,0.5721004793377098*pi) q[28];
U1q(0.570733955853459*pi,0.02824701256249984*pi) q[29];
U1q(0.751928339067812*pi,1.9291845986493286*pi) q[30];
U1q(0.439748244728187*pi,0.6324559468843702*pi) q[31];
U1q(0.365502749007772*pi,0.9696022192380003*pi) q[32];
U1q(0.330263416887001*pi,0.19566062089900527*pi) q[33];
U1q(0.312586328312346*pi,1.9574651242884205*pi) q[34];
U1q(0.703870237520719*pi,0.3310924407303313*pi) q[35];
U1q(0.232065423256679*pi,1.0278446167984647*pi) q[36];
U1q(0.674907126069862*pi,0.027165199175662735*pi) q[37];
U1q(0.378284795951526*pi,0.15563267932229863*pi) q[38];
U1q(0.476087796062846*pi,0.7398722883430704*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[14];
RZZ(0.5*pi) q[36],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[5],q[32];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[12],q[7];
RZZ(0.5*pi) q[8],q[13];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[29],q[17];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[22],q[34];
RZZ(0.5*pi) q[24],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[28],q[37];
RZZ(0.5*pi) q[30],q[33];
U1q(0.712683775293904*pi,1.0044068639084065*pi) q[0];
U1q(0.720286658139922*pi,1.44068984414014*pi) q[1];
U1q(0.708411742197461*pi,1.99397672454214*pi) q[2];
U1q(0.511150083828577*pi,0.1430608572169998*pi) q[3];
U1q(0.606670690617802*pi,1.2137851297228224*pi) q[4];
U1q(0.346760139238487*pi,0.7997232746322993*pi) q[5];
U1q(0.465415333933195*pi,0.5190742680157197*pi) q[6];
U1q(0.683908596666216*pi,0.6204306199278893*pi) q[7];
U1q(0.663367101750209*pi,0.5124781317247304*pi) q[8];
U1q(0.140865333983283*pi,1.7615372822450315*pi) q[9];
U1q(0.799050370220863*pi,0.9781808295671048*pi) q[10];
U1q(0.380595221921746*pi,0.9465163290305174*pi) q[11];
U1q(0.178106598525918*pi,1.2859775917167209*pi) q[12];
U1q(0.706121186043716*pi,1.5713032566306993*pi) q[13];
U1q(0.269694495888844*pi,0.8598457507284678*pi) q[14];
U1q(0.516707886889372*pi,0.13537202555198746*pi) q[15];
U1q(0.36665899703784*pi,0.5942486120583741*pi) q[16];
U1q(0.263693383006911*pi,0.7729802775140104*pi) q[17];
U1q(0.301320975476747*pi,0.7055109502411003*pi) q[18];
U1q(0.41801729491784*pi,1.1659765377554514*pi) q[19];
U1q(0.48033233711832*pi,0.42244188232661095*pi) q[20];
U1q(0.429615272143648*pi,0.6014358311067696*pi) q[21];
U1q(0.518734127001252*pi,1.6887382532262016*pi) q[22];
U1q(0.841241036983091*pi,1.39893443775434*pi) q[23];
U1q(0.198620652997853*pi,1.3738318972691896*pi) q[24];
U1q(0.491578000121798*pi,0.31324316605888214*pi) q[25];
U1q(0.653838660952955*pi,1.3607256947932669*pi) q[26];
U1q(0.618717431892547*pi,0.1551103740086628*pi) q[27];
U1q(0.639788037685311*pi,0.9294423167235202*pi) q[28];
U1q(0.511237534291126*pi,1.4114412516591006*pi) q[29];
U1q(0.232151363080637*pi,1.4947208183155993*pi) q[30];
U1q(0.702847846620315*pi,1.6488736700818691*pi) q[31];
U1q(0.600774705834746*pi,0.9304150828180502*pi) q[32];
U1q(0.37339386777608*pi,0.7788235352220552*pi) q[33];
U1q(0.863592949178326*pi,1.0669968472806008*pi) q[34];
U1q(0.322894529344145*pi,0.42463054939288103*pi) q[35];
U1q(0.404580381478791*pi,0.43477952194974456*pi) q[36];
U1q(0.792351287680736*pi,0.8959756266358543*pi) q[37];
U1q(0.137313664404306*pi,1.9030587739713685*pi) q[38];
U1q(0.691190352589073*pi,0.18314598775689994*pi) q[39];
rz(0.8733462449314731*pi) q[0];
rz(0.9460117614075401*pi) q[1];
rz(0.12043066083244014*pi) q[2];
rz(0.26809928560030016*pi) q[3];
rz(3.421010174764678*pi) q[4];
rz(1.4915861048741892*pi) q[5];
rz(0.04942937154043037*pi) q[6];
rz(1.0199450909540708*pi) q[7];
rz(1.8464586971302595*pi) q[8];
rz(1.2675838688723786*pi) q[9];
rz(2.915532755434735*pi) q[10];
rz(0.19549984850698365*pi) q[11];
rz(2.1269430096155*pi) q[12];
rz(2.4263047678561005*pi) q[13];
rz(2.944755823620902*pi) q[14];
rz(3.461986083550503*pi) q[15];
rz(0.5182661716495272*pi) q[16];
rz(0.7711873859420493*pi) q[17];
rz(0.9006713577396006*pi) q[18];
rz(0.7266468492756175*pi) q[19];
rz(0.9318618614139105*pi) q[20];
rz(0.012255653207690287*pi) q[21];
rz(1.5062059296931292*pi) q[22];
rz(0.79717649492411*pi) q[23];
rz(2.35152849444051*pi) q[24];
rz(2.056914116747178*pi) q[25];
rz(2.838237746880134*pi) q[26];
rz(2.0121351449958365*pi) q[27];
rz(1.3901799231496703*pi) q[28];
rz(2.7590860267159005*pi) q[29];
rz(1.0014855423328513*pi) q[30];
rz(0.05784917494422004*pi) q[31];
rz(3.23731048339602*pi) q[32];
rz(0.07106680457062353*pi) q[33];
rz(0.86607841441562*pi) q[34];
rz(0.07804380641664821*pi) q[35];
rz(2.843894303924035*pi) q[36];
rz(2.126603108088297*pi) q[37];
rz(2.8643362161575414*pi) q[38];
rz(1.3154621005398006*pi) q[39];
U1q(0.712683775293904*pi,0.877753108839882*pi) q[0];
U1q(0.720286658139922*pi,1.386701605547682*pi) q[1];
U1q(0.708411742197461*pi,1.11440738537458*pi) q[2];
U1q(1.51115008382858*pi,1.4111601428173*pi) q[3];
U1q(0.606670690617802*pi,1.634795304487566*pi) q[4];
U1q(0.346760139238487*pi,1.29130937950649*pi) q[5];
U1q(0.465415333933195*pi,1.568503639556146*pi) q[6];
U1q(0.683908596666216*pi,0.640375710881956*pi) q[7];
U1q(1.66336710175021*pi,1.358936828855*pi) q[8];
U1q(3.140865333983282*pi,0.0291211511174296*pi) q[9];
U1q(0.799050370220863*pi,0.893713585001837*pi) q[10];
U1q(0.380595221921746*pi,0.142016177537567*pi) q[11];
U1q(0.178106598525918*pi,0.412920601332248*pi) q[12];
U1q(0.706121186043716*pi,0.997608024486764*pi) q[13];
U1q(1.26969449588884*pi,0.80460157434937*pi) q[14];
U1q(3.516707886889372*pi,0.597358109102498*pi) q[15];
U1q(1.36665899703784*pi,0.112514783707856*pi) q[16];
U1q(0.263693383006911*pi,0.54416766345606*pi) q[17];
U1q(0.301320975476747*pi,0.60618230798075*pi) q[18];
U1q(0.41801729491784*pi,0.89262338703107*pi) q[19];
U1q(0.48033233711832*pi,0.354303743740523*pi) q[20];
U1q(0.429615272143648*pi,1.61369148431446*pi) q[21];
U1q(0.518734127001252*pi,0.194944182919301*pi) q[22];
U1q(0.841241036983091*pi,1.19611093267845*pi) q[23];
U1q(0.198620652997853*pi,0.7253603917097*pi) q[24];
U1q(1.4915780001218*pi,1.370157282806052*pi) q[25];
U1q(0.653838660952955*pi,1.19896344167337*pi) q[26];
U1q(1.61871743189255*pi,1.1672455190045*pi) q[27];
U1q(0.639788037685311*pi,1.3196222398731932*pi) q[28];
U1q(1.51123753429113*pi,1.170527278375003*pi) q[29];
U1q(1.23215136308064*pi,1.496206360648443*pi) q[30];
U1q(0.702847846620315*pi,0.706722845026087*pi) q[31];
U1q(1.60077470583475*pi,1.16772556621408*pi) q[32];
U1q(0.37339386777608*pi,1.849890339792685*pi) q[33];
U1q(1.86359294917833*pi,0.933075261696219*pi) q[34];
U1q(1.32289452934414*pi,1.5026743558095301*pi) q[35];
U1q(1.40458038147879*pi,0.278673825873774*pi) q[36];
U1q(0.792351287680736*pi,0.0225787347241513*pi) q[37];
U1q(1.13731366440431*pi,1.767394990128913*pi) q[38];
U1q(1.69119035258907*pi,0.498608088296742*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[14];
RZZ(0.5*pi) q[36],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[5],q[32];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[12],q[7];
RZZ(0.5*pi) q[8],q[13];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[29],q[17];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[22],q[34];
RZZ(0.5*pi) q[24],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[28],q[37];
RZZ(0.5*pi) q[30],q[33];
U1q(0.0490003127194353*pi,1.830047553581318*pi) q[0];
U1q(0.260003285106841*pi,0.8616215250693*pi) q[1];
U1q(0.872732756754013*pi,1.35647527650158*pi) q[2];
U1q(3.710962202804102*pi,1.8517975268293525*pi) q[3];
U1q(1.22475235347074*pi,0.47586254478300005*pi) q[4];
U1q(0.38359869131729*pi,1.98818381177293*pi) q[5];
U1q(3.227758760792783*pi,0.9387278905361098*pi) q[6];
U1q(0.217711532663194*pi,0.9052699452870301*pi) q[7];
U1q(3.633215890885981*pi,0.1128967226209906*pi) q[8];
U1q(3.507849072470435*pi,0.6751284704175129*pi) q[9];
U1q(1.34749916757527*pi,0.15788087174242005*pi) q[10];
U1q(0.520979512279954*pi,1.72355501372952*pi) q[11];
U1q(0.726903531001162*pi,0.991706467092622*pi) q[12];
U1q(1.43975416669124*pi,0.395526135389324*pi) q[13];
U1q(1.61798792981369*pi,0.891243762023957*pi) q[14];
U1q(1.43590384882358*pi,0.9003212801551301*pi) q[15];
U1q(1.29918442406572*pi,1.385035821188219*pi) q[16];
U1q(1.56561424113323*pi,1.342515090192634*pi) q[17];
U1q(0.318676462577983*pi,0.4782933552777*pi) q[18];
U1q(0.531150575756974*pi,1.46300079304159*pi) q[19];
U1q(0.205965662457452*pi,1.111163504645989*pi) q[20];
U1q(0.679753656471057*pi,0.65329492269189*pi) q[21];
U1q(1.36999003704351*pi,1.85790058360557*pi) q[22];
U1q(1.82090384228756*pi,0.756532181462526*pi) q[23];
U1q(1.50666381169969*pi,1.0822467790446781*pi) q[24];
U1q(1.76656176038412*pi,0.9989411376564601*pi) q[25];
U1q(1.74663110446396*pi,1.14641984390768*pi) q[26];
U1q(1.35246339550748*pi,1.0872630512841803*pi) q[27];
U1q(1.80613565981111*pi,1.9622804024873801*pi) q[28];
U1q(3.57073395585346*pi,0.5537215174715736*pi) q[29];
U1q(3.248071660932187*pi,0.06174258031471047*pi) q[30];
U1q(1.43974824472819*pi,0.69030512182859*pi) q[31];
U1q(1.36550274900777*pi,1.1285384297941312*pi) q[32];
U1q(0.330263416887001*pi,1.2667274254696301*pi) q[33];
U1q(3.687413671687653*pi,1.0426069846883956*pi) q[34];
U1q(1.70387023752072*pi,1.5962124644720816*pi) q[35];
U1q(3.76793457674332*pi,0.6856087310250474*pi) q[36];
U1q(0.674907126069862*pi,0.15376830726395996*pi) q[37];
U1q(3.621715204048475*pi,1.5148210847779864*pi) q[38];
U1q(3.523912203937153*pi,0.9418817877106158*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[1],q[28];
RZZ(0.5*pi) q[2],q[8];
RZZ(0.5*pi) q[4],q[3];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[25],q[6];
RZZ(0.5*pi) q[36],q[7];
RZZ(0.5*pi) q[24],q[9];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[21],q[12];
RZZ(0.5*pi) q[30],q[14];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[20],q[17];
RZZ(0.5*pi) q[19],q[32];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[27],q[38];
RZZ(0.5*pi) q[29],q[33];
RZZ(0.5*pi) q[35],q[31];
U1q(1.9129864749391*pi,0.67284787273057*pi) q[0];
U1q(1.13490714028377*pi,0.9336913149205599*pi) q[1];
U1q(0.644335483916534*pi,1.303613561196944*pi) q[2];
U1q(3.494795269081644*pi,0.3040884929871428*pi) q[3];
U1q(1.47144090963995*pi,1.8168178545338236*pi) q[4];
U1q(0.64849842407387*pi,0.9721244328382102*pi) q[5];
U1q(3.629527544644997*pi,0.09210690060662952*pi) q[6];
U1q(0.583392447665085*pi,0.08706955873041*pi) q[7];
U1q(1.53421893707065*pi,0.9087978792365488*pi) q[8];
U1q(1.75709631583776*pi,0.5371612781193016*pi) q[9];
U1q(3.358570454091148*pi,1.2385797735179547*pi) q[10];
U1q(0.645953099619999*pi,0.2356683083459301*pi) q[11];
U1q(0.246210158989552*pi,1.2410382109069529*pi) q[12];
U1q(3.2226989722119193*pi,1.9836557414888367*pi) q[13];
U1q(3.275216315706205*pi,1.1290364434747575*pi) q[14];
U1q(0.419791462816206*pi,1.8757470024653722*pi) q[15];
U1q(0.903511354433318*pi,0.3354285178600801*pi) q[16];
U1q(1.33102985416094*pi,0.6703847963943984*pi) q[17];
U1q(1.53522023403873*pi,0.79737164752036*pi) q[18];
U1q(1.43848694697534*pi,0.4553558532356301*pi) q[19];
U1q(0.852924935487905*pi,0.3173541208844399*pi) q[20];
U1q(1.78328456097762*pi,0.6543978049230299*pi) q[21];
U1q(3.428796648637997*pi,0.5827756021450536*pi) q[22];
U1q(1.37299185292093*pi,1.2468087787532234*pi) q[23];
U1q(3.596485812980996*pi,1.4783328534721254*pi) q[24];
U1q(1.18529242559325*pi,0.8294828950247082*pi) q[25];
U1q(3.726664722854097*pi,1.8061924511328744*pi) q[26];
U1q(0.688777661146737*pi,1.2826626645662953*pi) q[27];
U1q(3.419378828356616*pi,1.3054860217060655*pi) q[28];
U1q(0.598533684074705*pi,1.7557449777494305*pi) q[29];
U1q(3.659184101688549*pi,1.9931059986613295*pi) q[30];
U1q(3.583936156884799*pi,1.7197271689265672*pi) q[31];
U1q(1.52487185569063*pi,1.2615713856589923*pi) q[32];
U1q(1.15465982902641*pi,1.7096317012554598*pi) q[33];
U1q(1.71604978106556*pi,0.2160746141584527*pi) q[34];
U1q(0.828293674635924*pi,0.3790379552311718*pi) q[35];
U1q(1.79620718827554*pi,1.690997590843505*pi) q[36];
U1q(0.222846466608176*pi,0.60254621786091*pi) q[37];
U1q(3.596585392692921*pi,0.9394872103054595*pi) q[38];
U1q(3.605267532201393*pi,1.8544941892288458*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[1],q[26];
RZZ(0.5*pi) q[2],q[33];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[30],q[4];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[7],q[38];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[11],q[34];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[20],q[13];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[19],q[17];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[22],q[31];
RZZ(0.5*pi) q[23],q[32];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[27],q[37];
U1q(3.311317319627857*pi,0.9588576635438641*pi) q[0];
U1q(3.9118025143772734*pi,0.5081185657115164*pi) q[1];
U1q(1.12576314397253*pi,1.04650707065259*pi) q[2];
U1q(1.3347140612204*pi,1.0911028308970314*pi) q[3];
U1q(0.0833390163307269*pi,0.5704401345204531*pi) q[4];
U1q(1.39308606085112*pi,0.09069143348559994*pi) q[5];
U1q(1.79854449730114*pi,0.26355694310959654*pi) q[6];
U1q(0.393227467191231*pi,0.7745806218446303*pi) q[7];
U1q(1.73247838638974*pi,0.14823448039502773*pi) q[8];
U1q(0.46047904698666*pi,0.991914205838742*pi) q[9];
U1q(3.076042139141798*pi,0.14861089411825024*pi) q[10];
U1q(0.171051025628983*pi,1.0202657526582204*pi) q[11];
U1q(0.563138941596323*pi,0.08714858923933*pi) q[12];
U1q(3.367017151310953*pi,1.3598217177113563*pi) q[13];
U1q(3.345900493831725*pi,0.3017797651386189*pi) q[14];
U1q(0.476975744835322*pi,1.985476514873812*pi) q[15];
U1q(1.10264533523274*pi,1.1127979652394329*pi) q[16];
U1q(0.31952937379813*pi,1.0689682749670104*pi) q[17];
U1q(1.68488873238507*pi,1.2160412831760308*pi) q[18];
U1q(3.862559575858624*pi,1.9063089908308504*pi) q[19];
U1q(0.598889978077403*pi,1.40719109141892*pi) q[20];
U1q(3.655488389627142*pi,0.6231612541451486*pi) q[21];
U1q(1.18333061266508*pi,0.20604049891711096*pi) q[22];
U1q(1.2339500651021*pi,1.8771526513953432*pi) q[23];
U1q(3.751623566969496*pi,0.6617857551420756*pi) q[24];
U1q(3.580127467084053*pi,0.05723638729032832*pi) q[25];
U1q(3.553481207817619*pi,1.9530276954689554*pi) q[26];
U1q(1.34427413695412*pi,0.27462214929966056*pi) q[27];
U1q(3.649864152963851*pi,0.8052972745011857*pi) q[28];
U1q(1.47185895378448*pi,0.4168230069601804*pi) q[29];
U1q(3.324235199068597*pi,0.3549449433027245*pi) q[30];
U1q(3.633795982245239*pi,0.9849598066733867*pi) q[31];
U1q(1.77165422980891*pi,1.3966805329398966*pi) q[32];
U1q(1.44176188622698*pi,1.157914309941825*pi) q[33];
U1q(1.31095073893657*pi,0.9891168619443036*pi) q[34];
U1q(0.161030576509607*pi,1.5000906943354417*pi) q[35];
U1q(1.5064893945209*pi,1.179151264288195*pi) q[36];
U1q(0.236449754437332*pi,1.0000605000345804*pi) q[37];
U1q(3.658935581468598*pi,1.0195308518439745*pi) q[38];
U1q(1.55065629955769*pi,1.1259919074436455*pi) q[39];
RZZ(0.5*pi) q[0],q[22];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[38],q[2];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[31],q[6];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[21],q[8];
RZZ(0.5*pi) q[39],q[9];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[15],q[12];
RZZ(0.5*pi) q[27],q[13];
RZZ(0.5*pi) q[35],q[14];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[17],q[30];
RZZ(0.5*pi) q[29],q[18];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[32],q[24];
RZZ(0.5*pi) q[28],q[33];
U1q(1.56284286396696*pi,1.689864372671688*pi) q[0];
U1q(1.57609317509132*pi,0.43616743699251925*pi) q[1];
U1q(3.41038047969844*pi,1.6676421413738507*pi) q[2];
U1q(0.4048544606538*pi,0.5696604184513516*pi) q[3];
U1q(0.135186431571781*pi,1.5764497667524937*pi) q[4];
U1q(1.56452012278821*pi,1.345721489707267*pi) q[5];
U1q(0.505511683040376*pi,1.393088073019796*pi) q[6];
U1q(0.538150167481349*pi,1.4598328867214896*pi) q[7];
U1q(1.82798417884666*pi,1.3523857101976424*pi) q[8];
U1q(0.616289541781675*pi,1.1531543140295621*pi) q[9];
U1q(0.291487942092677*pi,1.3576444061742103*pi) q[10];
U1q(0.650056730840577*pi,1.7774750233431398*pi) q[11];
U1q(0.581180661022703*pi,1.66577505782007*pi) q[12];
U1q(1.5705585066276*pi,0.10904506079027132*pi) q[13];
U1q(0.442820335371004*pi,0.5351515258833484*pi) q[14];
U1q(0.497313680535401*pi,1.7374358108181522*pi) q[15];
U1q(1.17670536162564*pi,1.648266444187634*pi) q[16];
U1q(0.828302697900893*pi,1.3723660981656756*pi) q[17];
U1q(0.199022540309589*pi,1.1586204121416106*pi) q[18];
U1q(1.27538873249152*pi,0.052241068827276305*pi) q[19];
U1q(0.316728479737945*pi,0.9906571538244302*pi) q[20];
U1q(1.42792262993602*pi,1.4746402165980799*pi) q[21];
U1q(0.525323031693769*pi,1.1145857234969112*pi) q[22];
U1q(1.47108148720585*pi,0.5760758805362522*pi) q[23];
U1q(1.71616416059734*pi,0.5126829785085301*pi) q[24];
U1q(3.5849013429117242*pi,1.0387784384265446*pi) q[25];
U1q(3.443913735629144*pi,1.8924372518658847*pi) q[26];
U1q(1.13142028352966*pi,0.21931328230795577*pi) q[27];
U1q(1.97812169026052*pi,0.05638478739032582*pi) q[28];
U1q(1.13911698688851*pi,1.977489774721397*pi) q[29];
U1q(1.53539002543189*pi,1.1461365068690355*pi) q[30];
U1q(1.24402883381311*pi,1.7607865463504373*pi) q[31];
U1q(0.350479382597176*pi,0.07508077240789568*pi) q[32];
U1q(0.167421055683222*pi,1.6647354974253652*pi) q[33];
U1q(1.75965053003487*pi,0.31332289328597285*pi) q[34];
U1q(0.826789782165189*pi,1.7877435979459815*pi) q[35];
U1q(1.56512550361914*pi,1.5671514716663166*pi) q[36];
U1q(0.183420308773219*pi,1.9975706966641598*pi) q[37];
U1q(1.61770287701785*pi,0.6332562958375796*pi) q[38];
U1q(0.777772534920656*pi,0.07051205418874495*pi) q[39];
rz(0.31013562732831196*pi) q[0];
rz(3.5638325630074807*pi) q[1];
rz(0.3323578586261494*pi) q[2];
rz(1.4303395815486484*pi) q[3];
rz(2.4235502332475063*pi) q[4];
rz(0.6542785102927331*pi) q[5];
rz(0.6069119269802039*pi) q[6];
rz(2.5401671132785104*pi) q[7];
rz(2.6476142898023576*pi) q[8];
rz(0.8468456859704379*pi) q[9];
rz(0.6423555938257897*pi) q[10];
rz(2.22252497665686*pi) q[11];
rz(2.33422494217993*pi) q[12];
rz(1.8909549392097287*pi) q[13];
rz(1.4648484741166516*pi) q[14];
rz(2.2625641891818478*pi) q[15];
rz(0.351733555812366*pi) q[16];
rz(2.6276339018343244*pi) q[17];
rz(2.8413795878583894*pi) q[18];
rz(1.9477589311727237*pi) q[19];
rz(3.00934284617557*pi) q[20];
rz(2.52535978340192*pi) q[21];
rz(0.8854142765030888*pi) q[22];
rz(1.4239241194637478*pi) q[23];
rz(1.48731702149147*pi) q[24];
rz(0.9612215615734554*pi) q[25];
rz(0.10756274813411537*pi) q[26];
rz(3.7806867176920442*pi) q[27];
rz(1.9436152126096742*pi) q[28];
rz(2.022510225278603*pi) q[29];
rz(2.8538634931309645*pi) q[30];
rz(0.2392134536495627*pi) q[31];
rz(3.9249192275921043*pi) q[32];
rz(2.335264502574635*pi) q[33];
rz(1.6866771067140272*pi) q[34];
rz(2.2122564020540185*pi) q[35];
rz(2.4328485283336834*pi) q[36];
rz(2.0024293033358402*pi) q[37];
rz(3.3667437041624204*pi) q[38];
rz(3.929487945811255*pi) q[39];
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
