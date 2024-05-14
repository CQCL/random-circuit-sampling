OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.129255857707032*pi,1.248623349537828*pi) q[0];
U1q(0.62526170855428*pi,1.885388843817337*pi) q[1];
U1q(0.38896290102622*pi,1.80554469569418*pi) q[2];
U1q(0.541117392280231*pi,0.173057610325586*pi) q[3];
U1q(0.336897708172911*pi,0.740176989010231*pi) q[4];
U1q(0.827717942169403*pi,1.442645535199397*pi) q[5];
U1q(0.0349342554944632*pi,1.490663215037931*pi) q[6];
U1q(0.319981300742812*pi,0.2686012615495901*pi) q[7];
U1q(0.587509086283369*pi,0.706311234273451*pi) q[8];
U1q(0.546891037349693*pi,1.096254538559982*pi) q[9];
U1q(0.199180795456009*pi,1.9794064020311044*pi) q[10];
U1q(0.541770692497971*pi,0.484489214609496*pi) q[11];
U1q(0.653773673492284*pi,1.22552916792937*pi) q[12];
U1q(0.806735853932709*pi,1.791023142252879*pi) q[13];
U1q(0.526992804454407*pi,0.20063536103135*pi) q[14];
U1q(0.153468391556263*pi,0.198174829742151*pi) q[15];
U1q(0.597912035951203*pi,1.09960055577233*pi) q[16];
U1q(0.335980673085997*pi,1.02075156183679*pi) q[17];
U1q(0.921621411684944*pi,0.701557611994718*pi) q[18];
U1q(0.628539870457662*pi,0.590197439308049*pi) q[19];
U1q(0.389711313434079*pi,1.5266784388191321*pi) q[20];
U1q(0.382344494006724*pi,0.370791594485691*pi) q[21];
U1q(0.771573570741559*pi,0.70342792440597*pi) q[22];
U1q(0.451082558259007*pi,0.7692576879300099*pi) q[23];
U1q(0.678906505350503*pi,0.22200316343673*pi) q[24];
U1q(0.204472338961291*pi,1.754750138270436*pi) q[25];
U1q(0.765108997741224*pi,1.3517562269988619*pi) q[26];
U1q(0.726949599055091*pi,0.224922005366576*pi) q[27];
U1q(0.465987918692483*pi,0.448291097600796*pi) q[28];
U1q(0.558001029154728*pi,0.401037643609485*pi) q[29];
U1q(0.929487969702258*pi,1.808448846309057*pi) q[30];
U1q(0.419089909669111*pi,1.904295432622895*pi) q[31];
U1q(0.377419987416096*pi,1.876204948641443*pi) q[32];
U1q(0.488208497099974*pi,1.255109428358665*pi) q[33];
U1q(0.578709654929749*pi,1.201177501739324*pi) q[34];
U1q(0.195533568225738*pi,1.654400977010259*pi) q[35];
U1q(0.421410602786774*pi,0.437002884967962*pi) q[36];
U1q(0.75102927013474*pi,1.00720956648118*pi) q[37];
U1q(0.326322454727236*pi,1.31738026411645*pi) q[38];
U1q(0.625448938527193*pi,0.295246840702062*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[19],q[2];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[4],q[28];
RZZ(0.5*pi) q[22],q[5];
RZZ(0.5*pi) q[6],q[34];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[18],q[8];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[13],q[12];
RZZ(0.5*pi) q[15],q[14];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[20],q[37];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[29],q[31];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[33],q[32];
RZZ(0.5*pi) q[36],q[39];
U1q(0.448652727908637*pi,0.33096064434022*pi) q[0];
U1q(0.574240975367051*pi,1.4158522217091*pi) q[1];
U1q(0.943866207438246*pi,0.36616791339132004*pi) q[2];
U1q(0.411184639466972*pi,1.5422235248575*pi) q[3];
U1q(0.287420467263896*pi,1.51745684688772*pi) q[4];
U1q(0.419206193384458*pi,0.6628138602877001*pi) q[5];
U1q(0.651860226647084*pi,1.26058374038363*pi) q[6];
U1q(0.36568242003631*pi,1.77325095318173*pi) q[7];
U1q(0.410363408372016*pi,0.5755595269638201*pi) q[8];
U1q(0.571378886918063*pi,1.6041806348708203*pi) q[9];
U1q(0.843891621827634*pi,1.9495586291231102*pi) q[10];
U1q(0.522213953916144*pi,1.2338739408923698*pi) q[11];
U1q(0.422776952116664*pi,0.0201357243373923*pi) q[12];
U1q(0.306931513213968*pi,0.9267044161894802*pi) q[13];
U1q(0.346423480539345*pi,0.33380772419664995*pi) q[14];
U1q(0.866640259192289*pi,1.7430374814566498*pi) q[15];
U1q(0.422428212757425*pi,0.7728006489510699*pi) q[16];
U1q(0.113677739610427*pi,0.003296570558269929*pi) q[17];
U1q(0.340770534767171*pi,1.577425780754578*pi) q[18];
U1q(0.454696156398536*pi,0.7463575515421901*pi) q[19];
U1q(0.122718505788487*pi,1.3826418008955899*pi) q[20];
U1q(0.378480061626351*pi,0.68111005126545*pi) q[21];
U1q(0.392353416100849*pi,0.3517585617216601*pi) q[22];
U1q(0.6390205838949*pi,1.5518168481168102*pi) q[23];
U1q(0.519465554592456*pi,1.0523581155306698*pi) q[24];
U1q(0.302393633807502*pi,0.8827337806151898*pi) q[25];
U1q(0.590460152312344*pi,1.1630034929912498*pi) q[26];
U1q(0.750675118887053*pi,1.019207100278297*pi) q[27];
U1q(0.75896221223246*pi,0.2859584392682599*pi) q[28];
U1q(0.455995125488006*pi,0.3144329325836599*pi) q[29];
U1q(0.535359597671801*pi,0.1754950021972801*pi) q[30];
U1q(0.881107544515477*pi,1.6539952157453799*pi) q[31];
U1q(0.725997478517555*pi,0.4056752941209001*pi) q[32];
U1q(0.6286593084608*pi,1.4807153176275198*pi) q[33];
U1q(0.42194243352926*pi,0.90320469697512*pi) q[34];
U1q(0.85501286222782*pi,1.41656017273744*pi) q[35];
U1q(0.24637271595591*pi,0.62216940450987*pi) q[36];
U1q(0.369578446859691*pi,0.18808144410442007*pi) q[37];
U1q(0.569306824400599*pi,0.59513093716276*pi) q[38];
U1q(0.362470622445208*pi,1.95051978965791*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[30],q[1];
RZZ(0.5*pi) q[20],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[33],q[5];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[39],q[16];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[25],q[21];
RZZ(0.5*pi) q[24],q[34];
RZZ(0.5*pi) q[36],q[27];
RZZ(0.5*pi) q[32],q[29];
RZZ(0.5*pi) q[38],q[37];
U1q(0.861289790690229*pi,1.1898655478022304*pi) q[0];
U1q(0.404717471763184*pi,0.8837661121084102*pi) q[1];
U1q(0.868396073070813*pi,1.3260411435366297*pi) q[2];
U1q(0.794961846059759*pi,1.6823547370407903*pi) q[3];
U1q(0.284524590699369*pi,1.9059507015964003*pi) q[4];
U1q(0.804070429299867*pi,0.37480444375909006*pi) q[5];
U1q(0.757559831286315*pi,0.23840378119639993*pi) q[6];
U1q(0.194966083788494*pi,1.0079130379397698*pi) q[7];
U1q(0.510192874890791*pi,0.42959213720111*pi) q[8];
U1q(0.464769980560841*pi,1.9936519406802997*pi) q[9];
U1q(0.863548269260769*pi,1.2079216675565103*pi) q[10];
U1q(0.197708120855512*pi,0.47129728849721975*pi) q[11];
U1q(0.349669017116952*pi,1.44603272746805*pi) q[12];
U1q(0.344753931293204*pi,1.7007651753958202*pi) q[13];
U1q(0.629171717024646*pi,0.6335739988527198*pi) q[14];
U1q(0.220954353692472*pi,0.8488944318708498*pi) q[15];
U1q(0.476029499981153*pi,0.51755214987688*pi) q[16];
U1q(0.780002901449348*pi,0.09188159768656012*pi) q[17];
U1q(0.853527913094565*pi,0.91022570906515*pi) q[18];
U1q(0.873384873221579*pi,1.6891936788185697*pi) q[19];
U1q(0.085481903242041*pi,0.6818682369210203*pi) q[20];
U1q(0.460508338486252*pi,0.6851616158905798*pi) q[21];
U1q(0.631003127908613*pi,1.5138740427301798*pi) q[22];
U1q(0.157841212262315*pi,0.8972643386322998*pi) q[23];
U1q(0.919660234745477*pi,0.8059929887999902*pi) q[24];
U1q(0.857672431095967*pi,1.61336639603367*pi) q[25];
U1q(0.82948066906858*pi,0.4145185414549002*pi) q[26];
U1q(0.532270023293869*pi,1.89127684099515*pi) q[27];
U1q(0.418207791820529*pi,0.5773641879554101*pi) q[28];
U1q(0.341706034307948*pi,1.2783136930248902*pi) q[29];
U1q(0.131406504676341*pi,0.8738210362361096*pi) q[30];
U1q(0.313706061671516*pi,1.0256453415038402*pi) q[31];
U1q(0.726470192922497*pi,1.6371770801886*pi) q[32];
U1q(0.654168203126386*pi,1.9483460318535304*pi) q[33];
U1q(0.757541987513438*pi,0.5259058430498298*pi) q[34];
U1q(0.29964097101771*pi,1.8931896114999303*pi) q[35];
U1q(0.376908799058867*pi,0.51062273167926*pi) q[36];
U1q(0.425358606151419*pi,1.18414350658235*pi) q[37];
U1q(0.494088309829775*pi,1.32206003313091*pi) q[38];
U1q(0.484746111860659*pi,1.0435805376186504*pi) q[39];
RZZ(0.5*pi) q[19],q[0];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[18],q[3];
RZZ(0.5*pi) q[4],q[39];
RZZ(0.5*pi) q[36],q[5];
RZZ(0.5*pi) q[20],q[6];
RZZ(0.5*pi) q[7],q[23];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[22],q[10];
RZZ(0.5*pi) q[11],q[16];
RZZ(0.5*pi) q[14],q[13];
RZZ(0.5*pi) q[15],q[26];
RZZ(0.5*pi) q[30],q[17];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[33],q[25];
RZZ(0.5*pi) q[32],q[27];
RZZ(0.5*pi) q[29],q[37];
RZZ(0.5*pi) q[38],q[34];
U1q(0.656544796235344*pi,1.40732019312905*pi) q[0];
U1q(0.535443771952019*pi,1.7559634505767798*pi) q[1];
U1q(0.744709123716443*pi,0.3376680466840698*pi) q[2];
U1q(0.240282997140011*pi,1.1943768719499204*pi) q[3];
U1q(0.39581053755134*pi,1.07337955946507*pi) q[4];
U1q(0.316706220704992*pi,0.22894551694263043*pi) q[5];
U1q(0.298607963146388*pi,0.7168471547748894*pi) q[6];
U1q(0.768915478148034*pi,0.9097261248347399*pi) q[7];
U1q(0.288580186061382*pi,1.5725733946795302*pi) q[8];
U1q(0.171068137003277*pi,0.5941572286305696*pi) q[9];
U1q(0.658417756400698*pi,1.9358961750593107*pi) q[10];
U1q(0.795780097794684*pi,1.4332236014226805*pi) q[11];
U1q(0.435544346565616*pi,0.4414758187443901*pi) q[12];
U1q(0.622981287647542*pi,1.6813131687983596*pi) q[13];
U1q(0.589563834597967*pi,0.7318157450962302*pi) q[14];
U1q(0.216560885770843*pi,1.4798886257337003*pi) q[15];
U1q(0.551190895937171*pi,0.7853859918752404*pi) q[16];
U1q(0.558945727369255*pi,1.37094258965349*pi) q[17];
U1q(0.347832155018465*pi,0.8771138603158999*pi) q[18];
U1q(0.295203388298188*pi,1.9090900087289597*pi) q[19];
U1q(0.443764652058008*pi,0.7724272833017798*pi) q[20];
U1q(0.42428505387263*pi,0.7396437139681602*pi) q[21];
U1q(0.571955334454868*pi,1.6858827177328397*pi) q[22];
U1q(0.32457965683328*pi,1.0036748626060898*pi) q[23];
U1q(0.85792102995371*pi,1.6413026909739*pi) q[24];
U1q(0.919113451875169*pi,1.3837873792449304*pi) q[25];
U1q(0.385580158273533*pi,0.39795857007084035*pi) q[26];
U1q(0.638263181647925*pi,1.5997026622970698*pi) q[27];
U1q(0.656268482960673*pi,1.4228701197809004*pi) q[28];
U1q(0.814503063866203*pi,0.9431713055780797*pi) q[29];
U1q(0.241672822754113*pi,0.23217519450810986*pi) q[30];
U1q(0.473746672099782*pi,0.99723098174616*pi) q[31];
U1q(0.734550492937708*pi,1.12752826988301*pi) q[32];
U1q(0.340800247181687*pi,0.5968031083815601*pi) q[33];
U1q(0.690544817356297*pi,0.17950325545998957*pi) q[34];
U1q(0.624914091908811*pi,0.24711532852356033*pi) q[35];
U1q(0.505899996670852*pi,1.0241367718031*pi) q[36];
U1q(0.501583227809969*pi,0.7054690960146299*pi) q[37];
U1q(0.721679691178668*pi,1.1120394502995108*pi) q[38];
U1q(0.434887765823849*pi,0.24147673043584028*pi) q[39];
RZZ(0.5*pi) q[23],q[0];
RZZ(0.5*pi) q[8],q[1];
RZZ(0.5*pi) q[2],q[21];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[32],q[6];
RZZ(0.5*pi) q[7],q[22];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[13],q[16];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[25],q[18];
RZZ(0.5*pi) q[20],q[34];
RZZ(0.5*pi) q[24],q[37];
RZZ(0.5*pi) q[28],q[38];
RZZ(0.5*pi) q[36],q[31];
RZZ(0.5*pi) q[33],q[39];
U1q(0.6194753882073*pi,1.4427770881157205*pi) q[0];
U1q(0.79192339550675*pi,1.92657638566852*pi) q[1];
U1q(0.823995945246025*pi,1.3126744030268007*pi) q[2];
U1q(0.608965983650496*pi,0.2279315142012699*pi) q[3];
U1q(0.474509936235023*pi,0.7966640335008606*pi) q[4];
U1q(0.719224752258669*pi,0.47456711788214*pi) q[5];
U1q(0.672767487505384*pi,0.1566357332454995*pi) q[6];
U1q(0.58827365086339*pi,0.7211403028496708*pi) q[7];
U1q(0.672160246311487*pi,1.8334687977771793*pi) q[8];
U1q(0.620815721968271*pi,1.6573806146554002*pi) q[9];
U1q(0.720326512588174*pi,1.4652417479684008*pi) q[10];
U1q(0.396379536496852*pi,1.6316329485272991*pi) q[11];
U1q(0.386446551258342*pi,1.9541263348785396*pi) q[12];
U1q(0.507279979786155*pi,1.8405642903015895*pi) q[13];
U1q(0.738783932471093*pi,0.4131557402125292*pi) q[14];
U1q(0.399734063408834*pi,0.12405503531780049*pi) q[15];
U1q(0.757825591510033*pi,1.3452757515585994*pi) q[16];
U1q(0.759665793603179*pi,1.5302275511246703*pi) q[17];
U1q(0.646402883482929*pi,0.9918973161988598*pi) q[18];
U1q(0.201615544569094*pi,1.6623647715759198*pi) q[19];
U1q(0.861472589151415*pi,1.1448689410221*pi) q[20];
U1q(0.499291465441637*pi,0.73304771924753*pi) q[21];
U1q(0.66160600710456*pi,1.8115978908576*pi) q[22];
U1q(0.433573787292175*pi,1.5000138017772002*pi) q[23];
U1q(0.795402503301663*pi,1.3075866444178992*pi) q[24];
U1q(0.332520397818674*pi,0.022584560708400758*pi) q[25];
U1q(0.447211738583961*pi,1.1027558187925397*pi) q[26];
U1q(0.265405267482165*pi,0.4437887178625801*pi) q[27];
U1q(0.565667436869197*pi,1.6861042376017608*pi) q[28];
U1q(0.489006665438946*pi,1.1355925846081796*pi) q[29];
U1q(0.65426693538931*pi,1.1001675243806996*pi) q[30];
U1q(0.465212604719357*pi,0.33703004885430055*pi) q[31];
U1q(0.555449669794381*pi,1.9698849800593896*pi) q[32];
U1q(0.492237634594392*pi,0.6557662685167394*pi) q[33];
U1q(0.846307188399344*pi,0.7969972263328797*pi) q[34];
U1q(0.264080784483809*pi,1.6158182291930991*pi) q[35];
U1q(0.215531077184526*pi,1.4487922284999009*pi) q[36];
U1q(0.344360538728885*pi,0.34965606176650965*pi) q[37];
U1q(0.847561842919514*pi,1.3965295993879003*pi) q[38];
U1q(0.40535527300066*pi,1.8680810311582992*pi) q[39];
RZZ(0.5*pi) q[15],q[0];
RZZ(0.5*pi) q[11],q[1];
RZZ(0.5*pi) q[2],q[38];
RZZ(0.5*pi) q[3],q[37];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[13];
RZZ(0.5*pi) q[8],q[17];
RZZ(0.5*pi) q[25],q[9];
RZZ(0.5*pi) q[31],q[10];
RZZ(0.5*pi) q[26],q[12];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[19],q[39];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[30],q[29];
RZZ(0.5*pi) q[33],q[36];
U1q(0.0747025370872795*pi,1.4213052580871999*pi) q[0];
U1q(0.283876768076624*pi,1.4617265338084007*pi) q[1];
U1q(0.366169643783702*pi,0.5216444890636005*pi) q[2];
U1q(0.844127740857902*pi,1.6262612571876005*pi) q[3];
U1q(0.422503219469327*pi,1.7951614856293006*pi) q[4];
U1q(0.453796342990776*pi,1.8469993447862993*pi) q[5];
U1q(0.374156618153369*pi,0.4525383480578*pi) q[6];
U1q(0.693400093384234*pi,0.7583802129135009*pi) q[7];
U1q(0.298064402033097*pi,0.9095290714002005*pi) q[8];
U1q(0.544650238106756*pi,0.7602176078921996*pi) q[9];
U1q(0.340859424167226*pi,0.8376942752957*pi) q[10];
U1q(0.419572813253224*pi,1.7155360316764998*pi) q[11];
U1q(0.437267216009698*pi,1.8789708619337002*pi) q[12];
U1q(0.527593531715477*pi,0.7837059018677905*pi) q[13];
U1q(0.180485435613113*pi,1.6876310439332993*pi) q[14];
U1q(0.29245352453178*pi,0.6490665133161002*pi) q[15];
U1q(0.384356499535906*pi,1.0210361940380004*pi) q[16];
U1q(0.103010957592834*pi,0.052006302406299554*pi) q[17];
U1q(0.631025250124397*pi,1.6517819518129597*pi) q[18];
U1q(0.909619041180578*pi,0.7368802106580699*pi) q[19];
U1q(0.796887379611386*pi,0.5526510783658996*pi) q[20];
U1q(0.376956591567289*pi,1.1075918230118003*pi) q[21];
U1q(0.262190738625567*pi,1.7362565409026*pi) q[22];
U1q(0.759018328112667*pi,1.0385676198393003*pi) q[23];
U1q(0.666971644940611*pi,0.034439859557799934*pi) q[24];
U1q(0.3663418732141*pi,0.7673322294662004*pi) q[25];
U1q(0.795758312560874*pi,1.0205605288123003*pi) q[26];
U1q(0.304694616891573*pi,1.2478605376150496*pi) q[27];
U1q(0.357234464728304*pi,0.44045599578591066*pi) q[28];
U1q(0.471204520281056*pi,0.2667871059095903*pi) q[29];
U1q(0.772134978360037*pi,0.5689940362188999*pi) q[30];
U1q(0.531647201644345*pi,0.5299114132130995*pi) q[31];
U1q(0.857965081869059*pi,1.2310162087311998*pi) q[32];
U1q(0.71882095282105*pi,1.0785836118120997*pi) q[33];
U1q(0.431344919134934*pi,0.05977800225988972*pi) q[34];
U1q(0.316500602337798*pi,0.1602739156518993*pi) q[35];
U1q(0.240255486888131*pi,1.0513295789594999*pi) q[36];
U1q(0.347633089800484*pi,0.7639624050049996*pi) q[37];
U1q(0.802199801970686*pi,0.07201351338540007*pi) q[38];
U1q(0.454169889451288*pi,1.7325350360712992*pi) q[39];
RZZ(0.5*pi) q[0],q[21];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[19],q[3];
RZZ(0.5*pi) q[8],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[29],q[6];
RZZ(0.5*pi) q[35],q[9];
RZZ(0.5*pi) q[28],q[10];
RZZ(0.5*pi) q[33],q[12];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[36],q[18];
RZZ(0.5*pi) q[20],q[24];
RZZ(0.5*pi) q[25],q[22];
RZZ(0.5*pi) q[39],q[27];
RZZ(0.5*pi) q[32],q[31];
U1q(0.154183655495507*pi,0.19128587670370045*pi) q[0];
U1q(0.472952264699822*pi,0.22591992009600048*pi) q[1];
U1q(0.797895478106816*pi,1.5965821546287984*pi) q[2];
U1q(0.395875563495488*pi,0.03342759153959918*pi) q[3];
U1q(0.289118897823139*pi,0.8934548037283996*pi) q[4];
U1q(0.417729914995892*pi,0.3851546525716998*pi) q[5];
U1q(0.582364669522676*pi,0.8652988768360004*pi) q[6];
U1q(0.434391674440933*pi,0.9573535258086991*pi) q[7];
U1q(0.294356946023336*pi,0.5815350335731004*pi) q[8];
U1q(0.761627874905284*pi,1.4278188709228008*pi) q[9];
U1q(0.245650425389107*pi,0.30553605180140053*pi) q[10];
U1q(0.535500225654862*pi,0.8518546838360983*pi) q[11];
U1q(0.347883082815992*pi,0.12310669301839994*pi) q[12];
U1q(0.294621222709581*pi,0.8525002935549999*pi) q[13];
U1q(0.664052905640458*pi,0.1263289547908002*pi) q[14];
U1q(0.630933305772909*pi,0.25208220513500024*pi) q[15];
U1q(0.741878316752149*pi,1.5639625630414002*pi) q[16];
U1q(0.346371109080522*pi,0.4695939039393995*pi) q[17];
U1q(0.539914291516515*pi,0.8736519723300802*pi) q[18];
U1q(0.101595766803413*pi,0.07027123808759939*pi) q[19];
U1q(0.0548753585484641*pi,1.2838713894901996*pi) q[20];
U1q(0.508663498044208*pi,1.2836178820605006*pi) q[21];
U1q(0.507769763146428*pi,1.562371612194699*pi) q[22];
U1q(0.902863390851013*pi,0.9749597030698993*pi) q[23];
U1q(0.657572427149467*pi,0.8691004486480001*pi) q[24];
U1q(0.586787313386323*pi,1.0371558428781995*pi) q[25];
U1q(0.519845032318312*pi,1.9817925390376008*pi) q[26];
U1q(0.321834440897115*pi,0.35448257689548957*pi) q[27];
U1q(0.418241534971102*pi,1.7531227146575006*pi) q[28];
U1q(0.231428955963339*pi,1.1201623335335*pi) q[29];
U1q(0.209300399957734*pi,1.0049086967334997*pi) q[30];
U1q(0.0545043613631666*pi,1.0625762195516*pi) q[31];
U1q(0.226846122069394*pi,1.1705997633848*pi) q[32];
U1q(0.397481812808277*pi,0.9709563100119993*pi) q[33];
U1q(0.786807127573776*pi,0.3913694824078*pi) q[34];
U1q(0.61199603837228*pi,1.8814971053222997*pi) q[35];
U1q(0.559597748484192*pi,0.4666508374348002*pi) q[36];
U1q(0.099118638616892*pi,1.3552976295350003*pi) q[37];
U1q(0.27515332487972*pi,1.5985119036967*pi) q[38];
U1q(0.233978583553854*pi,1.2761675417755*pi) q[39];
RZZ(0.5*pi) q[7],q[0];
RZZ(0.5*pi) q[1],q[34];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[36],q[3];
RZZ(0.5*pi) q[4],q[38];
RZZ(0.5*pi) q[25],q[5];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[8],q[39];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[17],q[10];
RZZ(0.5*pi) q[33],q[11];
RZZ(0.5*pi) q[32],q[12];
RZZ(0.5*pi) q[13],q[37];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[18],q[23];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[35],q[29];
U1q(0.348974705017713*pi,0.7404283260522995*pi) q[0];
U1q(0.539683659579762*pi,0.5512096459062992*pi) q[1];
U1q(0.917037386455583*pi,0.47307406883859926*pi) q[2];
U1q(0.60934223225558*pi,0.6903570247889004*pi) q[3];
U1q(0.407192354338304*pi,0.39980796326329937*pi) q[4];
U1q(0.523724590886542*pi,0.4613725126034005*pi) q[5];
U1q(0.442351929423091*pi,1.189575700335201*pi) q[6];
U1q(0.541363285517644*pi,0.08438582472800071*pi) q[7];
U1q(0.222753245646504*pi,1.275375742586899*pi) q[8];
U1q(0.861380251978881*pi,0.3764385941952*pi) q[9];
U1q(0.489242900379923*pi,1.287660470431799*pi) q[10];
U1q(0.570603023590026*pi,0.3925934226951995*pi) q[11];
U1q(0.563018826309709*pi,0.2618276785691993*pi) q[12];
U1q(0.5239594107407*pi,0.6322896399009004*pi) q[13];
U1q(0.183669325972971*pi,1.0057266572458001*pi) q[14];
U1q(0.810979671131356*pi,1.3236850643043994*pi) q[15];
U1q(0.497919305051423*pi,0.21127277767089936*pi) q[16];
U1q(0.392433631281576*pi,1.9849366573879994*pi) q[17];
U1q(0.514909492393278*pi,0.8839934381968693*pi) q[18];
U1q(0.591527431203875*pi,0.05964953670089912*pi) q[19];
U1q(0.237364960338845*pi,0.8040692977153014*pi) q[20];
U1q(0.479960198555*pi,0.8117707883201*pi) q[21];
U1q(0.38336823184182*pi,0.403215185351101*pi) q[22];
U1q(0.666610427060844*pi,0.2342561909853007*pi) q[23];
U1q(0.694871588928131*pi,0.9620215634362008*pi) q[24];
U1q(0.495903034829505*pi,1.5006267222766994*pi) q[25];
U1q(0.336727242593518*pi,0.22568199638740083*pi) q[26];
U1q(0.533663735064494*pi,0.5202062925196103*pi) q[27];
U1q(0.0309399911280261*pi,1.8436425363115987*pi) q[28];
U1q(0.611849991298407*pi,0.19195041219719933*pi) q[29];
U1q(0.0131219093884159*pi,0.9027952569375017*pi) q[30];
U1q(0.170365830518432*pi,1.4458705531761993*pi) q[31];
U1q(0.612298791137529*pi,1.8924168443724998*pi) q[32];
U1q(0.679539528922986*pi,0.33945747643790014*pi) q[33];
U1q(0.0806166204105695*pi,0.9079978558165998*pi) q[34];
U1q(0.912332609220153*pi,0.3561785443503993*pi) q[35];
U1q(0.790639680778354*pi,0.7544891855272002*pi) q[36];
U1q(0.38620938046831*pi,1.9184144412283999*pi) q[37];
U1q(0.613591324364511*pi,0.7474526448407008*pi) q[38];
U1q(0.142477258471466*pi,0.2245002958187996*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[7],q[18];
RZZ(0.5*pi) q[36],q[8];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[35],q[10];
RZZ(0.5*pi) q[11],q[34];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[19],q[13];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[24],q[16];
RZZ(0.5*pi) q[33],q[17];
RZZ(0.5*pi) q[29],q[20];
RZZ(0.5*pi) q[26],q[22];
RZZ(0.5*pi) q[25],q[23];
RZZ(0.5*pi) q[30],q[27];
U1q(0.119935257668838*pi,1.1777627421902999*pi) q[0];
U1q(0.350868914499451*pi,1.9801362730258987*pi) q[1];
U1q(0.843213538875305*pi,0.002543193889501083*pi) q[2];
U1q(0.938427997007217*pi,0.9572928821090017*pi) q[3];
U1q(0.17461339114216*pi,0.47687766806609844*pi) q[4];
U1q(0.530782763405761*pi,1.5167141371767983*pi) q[5];
U1q(0.86607040641393*pi,1.2938466373582003*pi) q[6];
U1q(0.398435473746266*pi,1.7667381517184992*pi) q[7];
U1q(0.235866449872912*pi,1.4796418016798007*pi) q[8];
U1q(0.59741596618369*pi,0.1925130012175984*pi) q[9];
U1q(0.668758512478752*pi,0.033797744553101694*pi) q[10];
U1q(0.517882928835127*pi,1.5476391312557993*pi) q[11];
U1q(0.68513888528258*pi,1.8937593113892994*pi) q[12];
U1q(0.70259480324783*pi,0.8318323892597004*pi) q[13];
U1q(0.127764789948554*pi,0.04446998053359863*pi) q[14];
U1q(0.482605785431795*pi,0.8758752185172014*pi) q[15];
U1q(0.33583541130327*pi,0.34483642256189917*pi) q[16];
U1q(0.834945928021586*pi,0.2232634609702*pi) q[17];
U1q(0.533910212660584*pi,0.8280357853100995*pi) q[18];
U1q(0.519129227100777*pi,0.4015030731998994*pi) q[19];
U1q(0.713219526686502*pi,1.7529156816458986*pi) q[20];
U1q(0.45098157393331*pi,1.1627393161574986*pi) q[21];
U1q(0.769771587413828*pi,1.9545536270080994*pi) q[22];
U1q(0.557304855039602*pi,0.7825537237503006*pi) q[23];
U1q(0.441382105959144*pi,0.023481800753998527*pi) q[24];
U1q(0.707689785202515*pi,0.025941713140500866*pi) q[25];
U1q(0.665041924854787*pi,0.4387284490370007*pi) q[26];
U1q(0.208556223178845*pi,1.0247634343035*pi) q[27];
U1q(0.647000402536591*pi,0.5473167576604006*pi) q[28];
U1q(0.379418521139701*pi,1.2589310796942996*pi) q[29];
U1q(0.337708708533889*pi,1.3335872791999996*pi) q[30];
U1q(0.410977214802225*pi,0.07329454355059895*pi) q[31];
U1q(0.416203638628209*pi,0.16601555971140058*pi) q[32];
U1q(0.434946334003231*pi,1.3451355845676005*pi) q[33];
U1q(0.303456555686675*pi,1.1720226423840998*pi) q[34];
U1q(0.815153758586447*pi,1.5854801277639012*pi) q[35];
U1q(0.53820067021742*pi,0.5708047004819008*pi) q[36];
U1q(0.7777941335264*pi,1.3145588217446011*pi) q[37];
U1q(0.538643469835854*pi,1.2179090897046017*pi) q[38];
U1q(0.641450262868368*pi,1.3227837009021997*pi) q[39];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[2],q[10];
RZZ(0.5*pi) q[3],q[34];
RZZ(0.5*pi) q[20],q[4];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[36],q[9];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[19],q[12];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[29],q[17];
RZZ(0.5*pi) q[18],q[22];
RZZ(0.5*pi) q[35],q[21];
RZZ(0.5*pi) q[23],q[38];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[33],q[26];
RZZ(0.5*pi) q[31],q[39];
U1q(0.673663890210415*pi,1.7737114365707*pi) q[0];
U1q(0.824744122365806*pi,1.9688978393735006*pi) q[1];
U1q(0.514042885724427*pi,0.8211506720952997*pi) q[2];
U1q(0.256419874440965*pi,1.5974217856897006*pi) q[3];
U1q(0.403675472601715*pi,0.16154245004300094*pi) q[4];
U1q(0.927064026099432*pi,0.666227280607*pi) q[5];
U1q(0.439588865809965*pi,0.012380834523700202*pi) q[6];
U1q(0.684859457269126*pi,1.8818909659247005*pi) q[7];
U1q(0.310897241892044*pi,0.7078983690740017*pi) q[8];
U1q(0.738777854066711*pi,0.8246674821557995*pi) q[9];
U1q(0.494917544515113*pi,1.119889647937999*pi) q[10];
U1q(0.400596782926286*pi,1.3127189747746009*pi) q[11];
U1q(0.189152618306119*pi,0.8916418449462995*pi) q[12];
U1q(0.26541680048762*pi,1.762976549022401*pi) q[13];
U1q(0.188700023862811*pi,0.9451004002321*pi) q[14];
U1q(0.33457526457328*pi,1.6621604781619013*pi) q[15];
U1q(0.639039518215146*pi,1.2758634877574018*pi) q[16];
U1q(0.662162195291922*pi,1.8896390647634007*pi) q[17];
U1q(0.721333981065912*pi,1.3665262497278992*pi) q[18];
U1q(0.396150957325713*pi,1.8589316494620007*pi) q[19];
U1q(0.562291907680084*pi,0.2952915313058*pi) q[20];
U1q(0.948589439910715*pi,1.1674875150618007*pi) q[21];
U1q(0.256789108936291*pi,0.8141368050800999*pi) q[22];
U1q(0.373752379620522*pi,0.8851993729934016*pi) q[23];
U1q(0.637944103005304*pi,1.7234478774473985*pi) q[24];
U1q(0.577067179540457*pi,0.8754202275426017*pi) q[25];
U1q(0.6288387170707*pi,0.7498437211215006*pi) q[26];
U1q(0.237697400439595*pi,0.7722870363420995*pi) q[27];
U1q(0.369331717313509*pi,0.9586192077689013*pi) q[28];
U1q(0.782238207779229*pi,0.38241327007110115*pi) q[29];
U1q(0.646522910035259*pi,0.6600966506103987*pi) q[30];
U1q(0.391082744992838*pi,1.385054586071*pi) q[31];
U1q(0.178578578874736*pi,1.890452811107199*pi) q[32];
U1q(0.475583639005981*pi,1.039007120198601*pi) q[33];
U1q(0.553663316248007*pi,1.4959282022378986*pi) q[34];
U1q(0.403995433160074*pi,0.5038629498249989*pi) q[35];
U1q(0.206912796735863*pi,0.5579169512154003*pi) q[36];
U1q(0.468298583524842*pi,1.9772305902389*pi) q[37];
U1q(0.484464130904121*pi,1.472390417577401*pi) q[38];
U1q(0.150056032048794*pi,0.0822129166520007*pi) q[39];
RZZ(0.5*pi) q[0],q[38];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[5],q[4];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[33],q[7];
RZZ(0.5*pi) q[8],q[27];
RZZ(0.5*pi) q[26],q[9];
RZZ(0.5*pi) q[39],q[10];
RZZ(0.5*pi) q[11],q[12];
RZZ(0.5*pi) q[13],q[34];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[19],q[16];
RZZ(0.5*pi) q[17],q[24];
RZZ(0.5*pi) q[18],q[29];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[36],q[21];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[35],q[28];
RZZ(0.5*pi) q[31],q[37];
U1q(0.571578688442217*pi,0.2840612967466001*pi) q[0];
U1q(0.569243063494036*pi,1.8277167378080996*pi) q[1];
U1q(0.605652752238067*pi,0.11080596846050028*pi) q[2];
U1q(0.16875721444226*pi,1.8506043750222005*pi) q[3];
U1q(0.873830634708647*pi,0.7158801102836989*pi) q[4];
U1q(0.518993003723077*pi,0.02629079122860034*pi) q[5];
U1q(0.202689515269076*pi,1.0141042968082985*pi) q[6];
U1q(0.343123877441796*pi,0.8925421582827013*pi) q[7];
U1q(0.188290675674143*pi,1.855441895054799*pi) q[8];
U1q(0.647988887616226*pi,0.18100223819210015*pi) q[9];
U1q(0.338420412200623*pi,1.686329526750999*pi) q[10];
U1q(0.698086428578837*pi,1.9505789140435006*pi) q[11];
U1q(0.297623217065771*pi,1.4547908943395989*pi) q[12];
U1q(0.428430900020802*pi,0.4437474539160995*pi) q[13];
U1q(0.660172713558065*pi,0.5655882790718998*pi) q[14];
U1q(0.810063002929372*pi,1.833884497188901*pi) q[15];
U1q(0.295659728954649*pi,1.3713716845721997*pi) q[16];
U1q(0.425844721180068*pi,1.2450010507238005*pi) q[17];
U1q(0.63879574802462*pi,0.03742441571280075*pi) q[18];
U1q(0.389868840780012*pi,1.4223093867638*pi) q[19];
U1q(0.668488949180463*pi,1.120326462749599*pi) q[20];
U1q(0.263105907723176*pi,0.8487805739998997*pi) q[21];
U1q(0.0904220228915233*pi,1.6792760064679015*pi) q[22];
U1q(0.872670208669604*pi,0.262165836447501*pi) q[23];
U1q(0.568907265106244*pi,0.9016107505760012*pi) q[24];
U1q(0.521531548172451*pi,0.9532150453058001*pi) q[25];
U1q(0.404977145190631*pi,1.0277914515993984*pi) q[26];
U1q(0.43093041502703*pi,1.998753977504201*pi) q[27];
U1q(0.574184751841196*pi,0.13967537296869992*pi) q[28];
U1q(0.243063009585623*pi,1.0788134008389*pi) q[29];
U1q(0.458492681129551*pi,0.6049774979204017*pi) q[30];
U1q(0.683261862244511*pi,1.0897273581548*pi) q[31];
U1q(0.423868955561541*pi,0.46976637600370097*pi) q[32];
U1q(0.835254344709273*pi,1.4047610423200005*pi) q[33];
U1q(0.834826045595804*pi,1.1491127228023004*pi) q[34];
U1q(0.531811371144151*pi,0.641557580650101*pi) q[35];
U1q(0.927952631900887*pi,0.6721445016709993*pi) q[36];
U1q(0.564957229295193*pi,1.7771714137889987*pi) q[37];
U1q(0.696725076130155*pi,0.6590404158134007*pi) q[38];
U1q(0.70117874776458*pi,0.6696207684918996*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[22],q[2];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[8],q[21];
RZZ(0.5*pi) q[9],q[13];
RZZ(0.5*pi) q[14],q[11];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[36],q[20];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[32],q[28];
RZZ(0.5*pi) q[33],q[29];
RZZ(0.5*pi) q[30],q[37];
U1q(0.808131640024403*pi,0.3390156116932985*pi) q[0];
U1q(0.615524915395424*pi,0.06481029742269939*pi) q[1];
U1q(0.67036217186464*pi,0.26147325792869935*pi) q[2];
U1q(0.847210622189344*pi,1.3054478389810988*pi) q[3];
U1q(0.612792400397216*pi,0.6215422741057992*pi) q[4];
U1q(0.316364023099072*pi,1.6442344678446013*pi) q[5];
U1q(0.417329630765304*pi,1.182392670045001*pi) q[6];
U1q(0.766322829314022*pi,1.6890004772387002*pi) q[7];
U1q(0.189547141507376*pi,0.5326627182006014*pi) q[8];
U1q(0.574309626854238*pi,0.9396762687447016*pi) q[9];
U1q(0.402999685843846*pi,0.6450486714802004*pi) q[10];
U1q(0.647454269024141*pi,1.6963193738791986*pi) q[11];
U1q(0.389872157538781*pi,0.16557267194909997*pi) q[12];
U1q(0.737015620760505*pi,1.3983352198784011*pi) q[13];
U1q(0.450513764060225*pi,1.9645108310833983*pi) q[14];
U1q(0.552377000362134*pi,1.7581980837152003*pi) q[15];
U1q(0.330798823830639*pi,1.6367948732780988*pi) q[16];
U1q(0.445365436527633*pi,1.4255066349691994*pi) q[17];
U1q(0.320683694453404*pi,1.5391249354156002*pi) q[18];
U1q(0.435109644592551*pi,1.0859306447518016*pi) q[19];
U1q(0.414295215597123*pi,1.6232533053635017*pi) q[20];
U1q(0.515148418595573*pi,0.5634154443968988*pi) q[21];
U1q(0.744330829332761*pi,0.9046943786851998*pi) q[22];
U1q(0.275314700521214*pi,1.1601542948959995*pi) q[23];
U1q(0.659824612042855*pi,0.21028720543829849*pi) q[24];
U1q(0.495979156971631*pi,1.6786317040998995*pi) q[25];
U1q(0.470918503846931*pi,1.1397875316066006*pi) q[26];
U1q(0.629325584873143*pi,0.05277525844299902*pi) q[27];
U1q(0.624269719095621*pi,0.6400371519187011*pi) q[28];
U1q(0.617787808659576*pi,1.340203590446599*pi) q[29];
U1q(0.680485199156758*pi,0.7612017287782997*pi) q[30];
U1q(0.301862082437102*pi,0.8640205377592984*pi) q[31];
U1q(0.843282178340879*pi,1.4764854997503*pi) q[32];
U1q(0.611957620430297*pi,0.7271758716257999*pi) q[33];
U1q(0.511965329674093*pi,1.1603339976365987*pi) q[34];
U1q(0.845361766691827*pi,1.1370837916338985*pi) q[35];
U1q(0.186525754705958*pi,1.0404506767741992*pi) q[36];
U1q(0.302591241084575*pi,1.9878408772790017*pi) q[37];
U1q(0.350858786797801*pi,1.9834021631292984*pi) q[38];
U1q(0.846135157916338*pi,0.1383448780681995*pi) q[39];
RZZ(0.5*pi) q[35],q[0];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[9],q[3];
RZZ(0.5*pi) q[4],q[37];
RZZ(0.5*pi) q[5],q[17];
RZZ(0.5*pi) q[30],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[10],q[38];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[26],q[13];
RZZ(0.5*pi) q[15],q[29];
RZZ(0.5*pi) q[23],q[16];
RZZ(0.5*pi) q[18],q[34];
RZZ(0.5*pi) q[36],q[19];
RZZ(0.5*pi) q[20],q[27];
RZZ(0.5*pi) q[28],q[21];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[25],q[32];
U1q(0.971349132883528*pi,1.5230566260201996*pi) q[0];
U1q(0.294168268641731*pi,0.512268525684501*pi) q[1];
U1q(0.490433298824842*pi,1.6056943523279017*pi) q[2];
U1q(0.663693299428814*pi,0.7652026671429013*pi) q[3];
U1q(0.539233905973264*pi,0.680988073669301*pi) q[4];
U1q(0.323700952808643*pi,0.9375006123430012*pi) q[5];
U1q(0.61515423772845*pi,0.7710654969080011*pi) q[6];
U1q(0.564475072469586*pi,0.10450380786490143*pi) q[7];
U1q(0.459070453852705*pi,1.2318835229734013*pi) q[8];
U1q(0.423102066013325*pi,1.7682483295534013*pi) q[9];
U1q(0.77725282244015*pi,0.07373375913369884*pi) q[10];
U1q(0.0780061023939362*pi,0.4528155616840017*pi) q[11];
U1q(0.094241355543708*pi,0.024380466916699817*pi) q[12];
U1q(0.267304321493013*pi,0.6451536965577986*pi) q[13];
U1q(0.495829730116829*pi,0.5101307901902992*pi) q[14];
U1q(0.289974022078951*pi,1.3352802801985*pi) q[15];
U1q(0.377966900599905*pi,1.9367711664858014*pi) q[16];
U1q(0.408344991253741*pi,1.0195022866266008*pi) q[17];
U1q(0.217188115466164*pi,1.3034903158104*pi) q[18];
U1q(0.67680849069215*pi,1.2867111740915007*pi) q[19];
U1q(0.801104938934661*pi,1.8605534030719006*pi) q[20];
U1q(0.658815098356453*pi,1.0886487112131995*pi) q[21];
U1q(0.413268113244454*pi,1.2782215242902986*pi) q[22];
U1q(0.733756414756681*pi,0.350727151091899*pi) q[23];
U1q(0.505949386529444*pi,0.2938421068595005*pi) q[24];
U1q(0.246520541490279*pi,0.42990670808649867*pi) q[25];
U1q(0.708007413057701*pi,1.1793173881243995*pi) q[26];
U1q(0.329326801581564*pi,0.8585135466523006*pi) q[27];
U1q(0.400929482271405*pi,1.9879578227319996*pi) q[28];
U1q(0.790826971642285*pi,0.07193601326899923*pi) q[29];
U1q(0.321910095629132*pi,0.5836529291097001*pi) q[30];
U1q(0.3738424674724*pi,0.336233603361201*pi) q[31];
U1q(0.81287163411043*pi,0.034819829285900994*pi) q[32];
U1q(0.686977368510647*pi,0.5337744324375997*pi) q[33];
U1q(0.455610984780755*pi,0.38306156186839857*pi) q[34];
U1q(0.240950041397471*pi,1.3226117062946017*pi) q[35];
U1q(0.32839926727048*pi,1.7020770903159992*pi) q[36];
U1q(0.764663479561155*pi,0.3771868115769017*pi) q[37];
U1q(0.643439486716173*pi,0.414563529617201*pi) q[38];
U1q(0.566770814407416*pi,1.1282292225671*pi) q[39];
rz(0.6310236172468997*pi) q[0];
rz(3.9041971591622016*pi) q[1];
rz(3.9260217968329982*pi) q[2];
rz(3.0902681655374984*pi) q[3];
rz(2.4383499773980013*pi) q[4];
rz(2.812404108271501*pi) q[5];
rz(1.9450244803797005*pi) q[6];
rz(3.5326918818599005*pi) q[7];
rz(3.6560723610898016*pi) q[8];
rz(1.7192210915153012*pi) q[9];
rz(2.8932020663800984*pi) q[10];
rz(0.09909410332399915*pi) q[11];
rz(3.452319121098*pi) q[12];
rz(1.7660352819220009*pi) q[13];
rz(1.4761340042336997*pi) q[14];
rz(3.4814763186489017*pi) q[15];
rz(0.6420969618020003*pi) q[16];
rz(1.3474703292182006*pi) q[17];
rz(3.8901152851909018*pi) q[18];
rz(1.1873513750431997*pi) q[19];
rz(2.634419442362301*pi) q[20];
rz(0.5975273464745001*pi) q[21];
rz(2.196381908831601*pi) q[22];
rz(0.17975049656159925*pi) q[23];
rz(3.7564694676383006*pi) q[24];
rz(1.8486441286745006*pi) q[25];
rz(3.8251414847687997*pi) q[26];
rz(0.10372785088220127*pi) q[27];
rz(3.4261220569235995*pi) q[28];
rz(2.072070472337*pi) q[29];
rz(3.845602526247699*pi) q[30];
rz(1.3553298463076011*pi) q[31];
rz(1.9593027959707996*pi) q[32];
rz(1.5977989086883007*pi) q[33];
rz(2.4358309977749997*pi) q[34];
rz(1.4536366592866017*pi) q[35];
rz(1.295316554500399*pi) q[36];
rz(3.3541087466023995*pi) q[37];
rz(2.943057018849*pi) q[38];
rz(0.14205154027870037*pi) q[39];
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
