OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(1.42020227879469*pi,0.9954711584006808*pi) q[0];
U1q(1.61558033419636*pi,1.8006380683919732*pi) q[1];
U1q(0.554698020517341*pi,0.546357760419289*pi) q[2];
U1q(0.547238010918721*pi,0.894767671500942*pi) q[3];
U1q(0.507289802263416*pi,0.992116892878572*pi) q[4];
U1q(1.30331626365632*pi,1.8249965182669867*pi) q[5];
U1q(0.560842227205746*pi,1.0672537297592*pi) q[6];
U1q(1.96469372282414*pi,1.8555015777728303*pi) q[7];
U1q(0.123417730739894*pi,1.788266406367695*pi) q[8];
U1q(0.613128213239225*pi,1.43051700234084*pi) q[9];
U1q(0.668965387028013*pi,1.53674477327986*pi) q[10];
U1q(1.41325040665886*pi,0.2285655722891868*pi) q[11];
U1q(1.50938838135886*pi,1.5173389812452651*pi) q[12];
U1q(0.42045931350913*pi,1.255867924379149*pi) q[13];
U1q(1.18966763828387*pi,0.5020564494236455*pi) q[14];
U1q(0.221681467400172*pi,1.135694751853721*pi) q[15];
U1q(3.698707300830922*pi,0.7766832286454223*pi) q[16];
U1q(3.709349432451452*pi,0.8635097447167088*pi) q[17];
U1q(0.631561911376063*pi,1.10650936582548*pi) q[18];
U1q(1.30100841808077*pi,0.780663270779985*pi) q[19];
U1q(1.40953027603767*pi,0.886026531551052*pi) q[20];
U1q(0.525941871236339*pi,0.0580783433226371*pi) q[21];
U1q(1.45168047921719*pi,1.4590431616373825*pi) q[22];
U1q(3.752245139351417*pi,0.61772338137305*pi) q[23];
U1q(0.707229925593341*pi,0.274898094970874*pi) q[24];
U1q(0.193892436315279*pi,1.4398514226723291*pi) q[25];
U1q(1.60448697826749*pi,1.3697316607994239*pi) q[26];
U1q(3.63403106976407*pi,0.5865325935532824*pi) q[27];
U1q(1.31855719148014*pi,1.3251792226725565*pi) q[28];
U1q(1.29673359397586*pi,0.891434026687243*pi) q[29];
U1q(0.756688068300818*pi,1.643064514020183*pi) q[30];
U1q(1.16450326236341*pi,1.2952063464875732*pi) q[31];
U1q(0.646831515949798*pi,1.350452528651899*pi) q[32];
U1q(1.734580507483*pi,1.8136255912132868*pi) q[33];
U1q(1.38315218724198*pi,0.892239749821182*pi) q[34];
U1q(0.492600624464133*pi,1.735443519668158*pi) q[35];
U1q(0.651304761708173*pi,0.763204902479188*pi) q[36];
U1q(1.76054204513646*pi,0.15579908686026514*pi) q[37];
U1q(3.548108266706861*pi,0.6382577744893141*pi) q[38];
U1q(1.30469433926821*pi,0.30444668724033985*pi) q[39];
RZZ(0.5*pi) q[27],q[0];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[6],q[30];
RZZ(0.5*pi) q[7],q[23];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[36],q[10];
RZZ(0.5*pi) q[12],q[11];
RZZ(0.5*pi) q[31],q[13];
RZZ(0.5*pi) q[14],q[16];
RZZ(0.5*pi) q[15],q[38];
RZZ(0.5*pi) q[17],q[29];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[37],q[25];
RZZ(0.5*pi) q[32],q[35];
U1q(0.638095185991976*pi,0.3091206178695268*pi) q[0];
U1q(0.853965498149334*pi,1.364536572909203*pi) q[1];
U1q(0.701978818016452*pi,0.78687620086901*pi) q[2];
U1q(0.726794463933067*pi,1.881925049570419*pi) q[3];
U1q(0.626924030636178*pi,1.552145059900947*pi) q[4];
U1q(0.579990117954535*pi,1.1545881421453768*pi) q[5];
U1q(0.354747939500317*pi,0.33737258127692993*pi) q[6];
U1q(0.657165601386963*pi,1.5743911937821702*pi) q[7];
U1q(0.554913409343629*pi,1.51704342011696*pi) q[8];
U1q(0.763362892307249*pi,0.9797574779275*pi) q[9];
U1q(0.333310164688642*pi,0.88330683229573*pi) q[10];
U1q(0.822558323633744*pi,0.990896935519817*pi) q[11];
U1q(0.386812326427926*pi,0.9694061368594253*pi) q[12];
U1q(0.250368190342193*pi,0.8958545779964102*pi) q[13];
U1q(0.700924952971263*pi,1.3224941551760954*pi) q[14];
U1q(0.444459034624498*pi,0.49878879862437*pi) q[15];
U1q(0.323135703995467*pi,0.7878073077713323*pi) q[16];
U1q(0.806098966040486*pi,1.9883770856349898*pi) q[17];
U1q(0.71633795855263*pi,1.2759218602364881*pi) q[18];
U1q(0.674582071289938*pi,0.565784247198019*pi) q[19];
U1q(0.409922553087705*pi,0.43264476326619206*pi) q[20];
U1q(0.857044862621926*pi,1.6353009835570802*pi) q[21];
U1q(0.785019158351967*pi,1.2916645757149525*pi) q[22];
U1q(0.871369352743652*pi,1.9077460310635401*pi) q[23];
U1q(0.184468273966276*pi,1.8771683945032*pi) q[24];
U1q(0.486428723840699*pi,0.4042389661277499*pi) q[25];
U1q(0.807978646305*pi,1.369343040548734*pi) q[26];
U1q(0.123258727824979*pi,0.5929827866804924*pi) q[27];
U1q(0.637650919824385*pi,0.9648188594597062*pi) q[28];
U1q(0.587190401189452*pi,1.854874153209943*pi) q[29];
U1q(0.659787351364424*pi,1.4113307205356*pi) q[30];
U1q(0.323194449607033*pi,1.219354472058733*pi) q[31];
U1q(0.410645598753438*pi,0.8305689377447498*pi) q[32];
U1q(0.721978627649*pi,0.07726033191883674*pi) q[33];
U1q(0.657806672624853*pi,1.8083663776793921*pi) q[34];
U1q(0.510343250545866*pi,0.8120751019953998*pi) q[35];
U1q(0.131427253477005*pi,1.4943753757925902*pi) q[36];
U1q(0.304428451042677*pi,0.7028263791396652*pi) q[37];
U1q(0.400755894845925*pi,1.438383221436224*pi) q[38];
U1q(0.681375318363423*pi,1.7064814747725099*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[2],q[18];
RZZ(0.5*pi) q[3],q[4];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[36],q[8];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[35];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[19],q[32];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[22],q[29];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[31],q[28];
RZZ(0.5*pi) q[30],q[34];
RZZ(0.5*pi) q[33],q[37];
U1q(0.360178720763905*pi,1.0171151100814009*pi) q[0];
U1q(0.474706255470897*pi,0.6733366603382733*pi) q[1];
U1q(0.557160288561759*pi,1.52304531407525*pi) q[2];
U1q(0.668544117368072*pi,0.88148015378316*pi) q[3];
U1q(0.778312951826834*pi,0.8592340763691202*pi) q[4];
U1q(0.790661278870819*pi,0.5057404087791566*pi) q[5];
U1q(0.61636459779852*pi,1.6630526017320797*pi) q[6];
U1q(0.343518330443199*pi,0.8374584912225407*pi) q[7];
U1q(0.0161455966114878*pi,1.8850144354626996*pi) q[8];
U1q(0.15836627244125*pi,0.33796785069602*pi) q[9];
U1q(0.700768181136185*pi,1.3380960147044698*pi) q[10];
U1q(0.115062123570894*pi,0.21082608887726684*pi) q[11];
U1q(0.482545186745929*pi,1.5316162137884248*pi) q[12];
U1q(0.377057921562136*pi,1.24363921191261*pi) q[13];
U1q(0.629623078325123*pi,1.2196197711188752*pi) q[14];
U1q(0.655291456202967*pi,1.7870672659741*pi) q[15];
U1q(0.0521637690520478*pi,1.0959357043457025*pi) q[16];
U1q(0.217259347274404*pi,1.128080928356769*pi) q[17];
U1q(0.326982174013816*pi,0.11771553695691983*pi) q[18];
U1q(0.132672378846647*pi,0.321297081815167*pi) q[19];
U1q(0.806571773624701*pi,1.5406289586909923*pi) q[20];
U1q(0.750662005796271*pi,1.9215523394272003*pi) q[21];
U1q(0.522682320520628*pi,1.5917224949315223*pi) q[22];
U1q(0.0170411457326908*pi,1.6826859344991396*pi) q[23];
U1q(0.40153358669097*pi,1.7330684274771198*pi) q[24];
U1q(0.422384021066834*pi,1.3472331064491296*pi) q[25];
U1q(0.160383596655299*pi,1.7318827189625337*pi) q[26];
U1q(0.615429881366941*pi,0.4697483879005224*pi) q[27];
U1q(0.779045707019585*pi,0.07676425303214618*pi) q[28];
U1q(0.576601584991436*pi,0.464887077509323*pi) q[29];
U1q(0.312198091418198*pi,0.26475577154934005*pi) q[30];
U1q(0.0438364738164473*pi,1.5969497026053627*pi) q[31];
U1q(0.396399238003022*pi,1.1800012943485303*pi) q[32];
U1q(0.294392891062673*pi,0.9408227613217268*pi) q[33];
U1q(0.446749221880749*pi,1.219278404836642*pi) q[34];
U1q(0.775742483980266*pi,1.7397243804064804*pi) q[35];
U1q(0.396657770527241*pi,0.4993366864939701*pi) q[36];
U1q(0.579174353695763*pi,1.0094889969534653*pi) q[37];
U1q(0.776807297939828*pi,1.2627817898106644*pi) q[38];
U1q(0.509042735261242*pi,1.17138184697495*pi) q[39];
RZZ(0.5*pi) q[24],q[0];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[5],q[4];
RZZ(0.5*pi) q[6],q[28];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[8],q[12];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[13],q[18];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[31],q[16];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[22],q[26];
RZZ(0.5*pi) q[37],q[29];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[33],q[36];
RZZ(0.5*pi) q[34],q[38];
U1q(0.227998560723241*pi,1.3022707745413502*pi) q[0];
U1q(0.218765674456192*pi,1.828470415712343*pi) q[1];
U1q(0.907796011091851*pi,1.1282417027876899*pi) q[2];
U1q(0.728900156779565*pi,1.5466294173270798*pi) q[3];
U1q(0.533707347402006*pi,0.09008860703082977*pi) q[4];
U1q(0.281669239049081*pi,1.1995593478650868*pi) q[5];
U1q(0.14331455826719*pi,0.7531910538507303*pi) q[6];
U1q(0.766550904877265*pi,0.10678205642941041*pi) q[7];
U1q(0.0681200929353996*pi,0.4212488763657003*pi) q[8];
U1q(0.7086615519103*pi,1.9201590161003903*pi) q[9];
U1q(0.374331796425996*pi,0.22218527594503001*pi) q[10];
U1q(0.476958044845605*pi,1.7425507603566563*pi) q[11];
U1q(0.370339104201921*pi,0.27587989006443436*pi) q[12];
U1q(0.165288425899225*pi,0.7319352656092803*pi) q[13];
U1q(0.833721985892704*pi,1.3310528472022654*pi) q[14];
U1q(0.282960069152581*pi,1.7560779891935105*pi) q[15];
U1q(0.60435155110745*pi,1.8387832815142415*pi) q[16];
U1q(0.449466722158518*pi,1.5241749464589596*pi) q[17];
U1q(0.0556333216142944*pi,1.2415402166473504*pi) q[18];
U1q(0.395280488914693*pi,0.16957206041749484*pi) q[19];
U1q(0.208463701037281*pi,0.8151040082912919*pi) q[20];
U1q(0.613602932390483*pi,1.1349525016473097*pi) q[21];
U1q(0.449690959345117*pi,0.28179486444358304*pi) q[22];
U1q(0.365113016373358*pi,1.5025169024675504*pi) q[23];
U1q(0.384717632578587*pi,0.3130003542849096*pi) q[24];
U1q(0.278106448998748*pi,0.4894875513200798*pi) q[25];
U1q(0.580342074892259*pi,0.8490405236955443*pi) q[26];
U1q(0.605146933969195*pi,0.3211319102218324*pi) q[27];
U1q(0.662689168098749*pi,1.7037723764834567*pi) q[28];
U1q(0.179075930010632*pi,0.9966128806122727*pi) q[29];
U1q(0.957619446951133*pi,1.96180074065781*pi) q[30];
U1q(0.640715759919855*pi,0.2788321151006832*pi) q[31];
U1q(0.208015006506008*pi,1.0032421657868298*pi) q[32];
U1q(0.313106381882197*pi,0.367864156732427*pi) q[33];
U1q(0.408892295031159*pi,1.7471317457652722*pi) q[34];
U1q(0.510381820631996*pi,0.25173708081651025*pi) q[35];
U1q(0.637250723420197*pi,1.5885008142392003*pi) q[36];
U1q(0.499797776647638*pi,1.4475710044161652*pi) q[37];
U1q(0.651935235864312*pi,1.4993557977459737*pi) q[38];
U1q(0.498314223532851*pi,1.64081176512307*pi) q[39];
RZZ(0.5*pi) q[29],q[0];
RZZ(0.5*pi) q[1],q[9];
RZZ(0.5*pi) q[7],q[2];
RZZ(0.5*pi) q[31],q[3];
RZZ(0.5*pi) q[4],q[28];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[39],q[8];
RZZ(0.5*pi) q[10],q[24];
RZZ(0.5*pi) q[11],q[38];
RZZ(0.5*pi) q[27],q[12];
RZZ(0.5*pi) q[33],q[13];
RZZ(0.5*pi) q[14],q[30];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[22],q[18];
RZZ(0.5*pi) q[21],q[20];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[34],q[35];
U1q(0.932619253371027*pi,1.729821541785201*pi) q[0];
U1q(0.690049473846226*pi,0.23526999172955332*pi) q[1];
U1q(0.693987404346671*pi,1.6463945290573596*pi) q[2];
U1q(0.308836580177496*pi,0.21058342300367983*pi) q[3];
U1q(0.244425996950416*pi,1.8125678154467604*pi) q[4];
U1q(0.222517438749087*pi,0.754361432394548*pi) q[5];
U1q(0.729578177694911*pi,1.5299978292568603*pi) q[6];
U1q(0.49374272042561*pi,1.6437967453454814*pi) q[7];
U1q(0.462967180899209*pi,1.9544149725025992*pi) q[8];
U1q(0.468690824554565*pi,1.50559667409701*pi) q[9];
U1q(0.498511196790046*pi,0.059591609184599825*pi) q[10];
U1q(0.620009911416595*pi,0.7772584786337466*pi) q[11];
U1q(0.676196499648001*pi,0.47233444608716546*pi) q[12];
U1q(0.752248627440266*pi,1.7039743672046992*pi) q[13];
U1q(0.740335934091868*pi,1.3176651962646853*pi) q[14];
U1q(0.711407386832899*pi,1.1145101205426098*pi) q[15];
U1q(0.697802147799292*pi,0.9943989619827516*pi) q[16];
U1q(0.77347402441269*pi,0.42727186628700764*pi) q[17];
U1q(0.753332003102894*pi,1.7474727063424496*pi) q[18];
U1q(0.510009561014276*pi,1.674221039859085*pi) q[19];
U1q(0.177800588277514*pi,0.6187606126533529*pi) q[20];
U1q(0.666641859878154*pi,0.10500998646397974*pi) q[21];
U1q(0.772962070571345*pi,0.15473605853957295*pi) q[22];
U1q(0.775884936161866*pi,0.021108155387119965*pi) q[23];
U1q(0.337095578006618*pi,1.6378762101390993*pi) q[24];
U1q(0.765284948516332*pi,0.6546222839035796*pi) q[25];
U1q(0.629772056754758*pi,1.1858243358208345*pi) q[26];
U1q(0.862180279815213*pi,0.047574314077011515*pi) q[27];
U1q(0.13331977095424*pi,1.862352770165936*pi) q[28];
U1q(0.755255569490009*pi,1.793328825678163*pi) q[29];
U1q(0.773930984537618*pi,0.2500127978673401*pi) q[30];
U1q(0.170535572765622*pi,1.5292382302391339*pi) q[31];
U1q(0.449068239847741*pi,1.4624112921808994*pi) q[32];
U1q(0.310843237794809*pi,0.3956653700510362*pi) q[33];
U1q(0.678026309754411*pi,1.0760301007926723*pi) q[34];
U1q(0.780791696414525*pi,1.37877381838552*pi) q[35];
U1q(0.518723360463873*pi,0.7756492727706892*pi) q[36];
U1q(0.173660041931596*pi,0.6220842403055746*pi) q[37];
U1q(0.540043538392807*pi,1.795828004313634*pi) q[38];
U1q(0.229013641355379*pi,0.07031387992097926*pi) q[39];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[8],q[20];
RZZ(0.5*pi) q[9],q[16];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[12],q[29];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[22],q[14];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[18],q[34];
RZZ(0.5*pi) q[36],q[21];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[37],q[30];
RZZ(0.5*pi) q[38],q[35];
U1q(0.969627518067417*pi,0.0422304532550708*pi) q[0];
U1q(0.522012973607475*pi,1.790770180366712*pi) q[1];
U1q(0.449173880085034*pi,1.6426512192644704*pi) q[2];
U1q(0.674469423060899*pi,1.73741035768398*pi) q[3];
U1q(0.521799978157716*pi,1.7635393661210994*pi) q[4];
U1q(0.705757354707097*pi,0.8348382773844865*pi) q[5];
U1q(0.272115803992218*pi,0.8849514191697008*pi) q[6];
U1q(0.915997112395763*pi,1.6085146403562316*pi) q[7];
U1q(0.799955018551722*pi,1.9227768454782996*pi) q[8];
U1q(0.387978703743518*pi,0.15025986466642038*pi) q[9];
U1q(0.195658384662775*pi,1.3773188860074708*pi) q[10];
U1q(0.952651756748911*pi,0.062777494418647*pi) q[11];
U1q(0.876177027319611*pi,0.05412154200816488*pi) q[12];
U1q(0.526808703865905*pi,1.9562110018183994*pi) q[13];
U1q(0.834623597135513*pi,1.584452770420036*pi) q[14];
U1q(0.227323474462536*pi,1.2288522842895997*pi) q[15];
U1q(0.575823928562012*pi,1.7622040595262227*pi) q[16];
U1q(0.444792128460461*pi,0.689553547169508*pi) q[17];
U1q(0.387081234375751*pi,0.6868252748516994*pi) q[18];
U1q(0.518426750247589*pi,1.4716977715557347*pi) q[19];
U1q(0.698031975499114*pi,0.5062913999357512*pi) q[20];
U1q(0.402103152646397*pi,0.3116499323055404*pi) q[21];
U1q(0.544533363761735*pi,1.2378709689879823*pi) q[22];
U1q(0.518241064906495*pi,1.5692008419100496*pi) q[23];
U1q(0.802054755671494*pi,1.5139333157276003*pi) q[24];
U1q(0.6362208689493*pi,1.50582911417791*pi) q[25];
U1q(0.452900425279289*pi,0.5371672219383239*pi) q[26];
U1q(0.56306510642289*pi,1.3748906330489739*pi) q[27];
U1q(0.413369402393446*pi,0.8867376330019567*pi) q[28];
U1q(0.655470719075153*pi,0.24482252689398276*pi) q[29];
U1q(0.851159115900539*pi,0.006440650520589841*pi) q[30];
U1q(0.6680298856099*pi,0.022270903031202494*pi) q[31];
U1q(0.316455406357688*pi,1.7881991215603001*pi) q[32];
U1q(0.832649304853407*pi,1.3093392931682857*pi) q[33];
U1q(0.436338561974063*pi,1.1634367275127317*pi) q[34];
U1q(0.846953343992689*pi,1.9694726253636006*pi) q[35];
U1q(0.210179033345704*pi,0.5551289446630001*pi) q[36];
U1q(0.373862539373868*pi,0.8578678517287148*pi) q[37];
U1q(0.839144951606204*pi,1.4029478021376143*pi) q[38];
U1q(0.714724299498217*pi,0.014993586936240533*pi) q[39];
RZZ(0.5*pi) q[36],q[0];
RZZ(0.5*pi) q[1],q[23];
RZZ(0.5*pi) q[2],q[28];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[5],q[35];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[29],q[11];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[22],q[37];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[32],q[38];
U1q(0.53782665018824*pi,1.2253498947042818*pi) q[0];
U1q(0.522786979780977*pi,0.20661265719780175*pi) q[1];
U1q(0.714869313560172*pi,1.4971209536431704*pi) q[2];
U1q(0.263870848077954*pi,0.56213835692565*pi) q[3];
U1q(0.40507354995314*pi,0.8709458121567994*pi) q[4];
U1q(0.932511859382513*pi,0.7203461412570871*pi) q[5];
U1q(0.486132057017105*pi,1.1041268452881994*pi) q[6];
U1q(0.204146681583132*pi,1.4586027685431304*pi) q[7];
U1q(0.728921899888012*pi,1.1241103559299006*pi) q[8];
U1q(0.65166967608231*pi,1.0321196083984*pi) q[9];
U1q(0.273766869959622*pi,1.4192750905524*pi) q[10];
U1q(0.630163995588764*pi,0.19105343775808592*pi) q[11];
U1q(0.744209268675153*pi,0.6965912147644673*pi) q[12];
U1q(0.662996304595787*pi,1.9483150379325007*pi) q[13];
U1q(0.672835531214822*pi,0.7457128374561464*pi) q[14];
U1q(0.794043248085125*pi,1.3403985604316997*pi) q[15];
U1q(0.600864576177457*pi,1.4743520203838223*pi) q[16];
U1q(0.466228239347931*pi,0.9839959710119075*pi) q[17];
U1q(0.70941482439998*pi,1.2648333108600998*pi) q[18];
U1q(0.442409153900685*pi,1.204611782716995*pi) q[19];
U1q(0.682308030034401*pi,1.3325603848606509*pi) q[20];
U1q(0.214622063421408*pi,0.42992423132059976*pi) q[21];
U1q(0.341332437343716*pi,0.33548263576678217*pi) q[22];
U1q(0.705888054164057*pi,0.050544624944450334*pi) q[23];
U1q(0.437784390973851*pi,1.1848662474946998*pi) q[24];
U1q(0.53681179260693*pi,0.7819709928877998*pi) q[25];
U1q(0.587144573740958*pi,1.3039968551449235*pi) q[26];
U1q(0.502532998035601*pi,1.3839455387219832*pi) q[27];
U1q(0.466913044426814*pi,0.4591426316980556*pi) q[28];
U1q(0.285552412846801*pi,1.9076337010452438*pi) q[29];
U1q(0.176397038859879*pi,0.24555843192290006*pi) q[30];
U1q(0.753968814579038*pi,0.33496149229167393*pi) q[31];
U1q(0.594545144841839*pi,1.6770097476860997*pi) q[32];
U1q(0.70760989923863*pi,1.968160003627986*pi) q[33];
U1q(0.667537222973326*pi,0.9436393139300812*pi) q[34];
U1q(0.642503075130094*pi,0.5620259564226*pi) q[35];
U1q(0.64158737742627*pi,0.5391393387607994*pi) q[36];
U1q(0.434871180624025*pi,0.7782145153853648*pi) q[37];
U1q(0.565597489998154*pi,0.7345052202353237*pi) q[38];
U1q(0.446214224601512*pi,1.47592538233814*pi) q[39];
rz(0.15632630369761813*pi) q[0];
rz(1.1763113129059182*pi) q[1];
rz(1.2382154537174106*pi) q[2];
rz(3.3757139671635503*pi) q[3];
rz(0.2883099635671993*pi) q[4];
rz(0.7425192756258134*pi) q[5];
rz(2.225945859309199*pi) q[6];
rz(0.04487739655506928*pi) q[7];
rz(0.8579126933069006*pi) q[8];
rz(2.187359230178*pi) q[9];
rz(2.7386141305213005*pi) q[10];
rz(2.641206185841913*pi) q[11];
rz(1.8606699192211344*pi) q[12];
rz(1.7163216575898996*pi) q[13];
rz(3.3479410074676537*pi) q[14];
rz(0.2806981570612006*pi) q[15];
rz(0.4264782379628773*pi) q[16];
rz(3.8997214940271903*pi) q[17];
rz(3.8439735939674993*pi) q[18];
rz(1.1996275393635845*pi) q[19];
rz(2.3490593484313482*pi) q[20];
rz(1.2515626363076002*pi) q[21];
rz(3.477453904791318*pi) q[22];
rz(2.448320515148449*pi) q[23];
rz(0.09113741314909873*pi) q[24];
rz(3.8674464930785994*pi) q[25];
rz(3.500893112167077*pi) q[26];
rz(3.7749507357968177*pi) q[27];
rz(2.9308638809481433*pi) q[28];
rz(2.2177745761391563*pi) q[29];
rz(0.5679803763100004*pi) q[30];
rz(3.532644334910927*pi) q[31];
rz(2.1082407902240003*pi) q[32];
rz(0.18616530108111284*pi) q[33];
rz(3.8395738700556166*pi) q[34];
rz(3.6645937475708994*pi) q[35];
rz(1.7073433091784*pi) q[36];
rz(0.32193155397913564*pi) q[37];
rz(2.3516785729923964*pi) q[38];
rz(1.1979500378109602*pi) q[39];
U1q(1.53782665018824*pi,0.381676198401924*pi) q[0];
U1q(0.522786979780977*pi,0.382923970103711*pi) q[1];
U1q(0.714869313560172*pi,1.735336407360578*pi) q[2];
U1q(3.263870848077954*pi,0.9378523240892*pi) q[3];
U1q(1.40507354995314*pi,0.159255775723987*pi) q[4];
U1q(1.93251185938251*pi,0.462865416882923*pi) q[5];
U1q(1.4861320570171*pi,0.330072704597444*pi) q[6];
U1q(1.20414668158313*pi,0.503480165098147*pi) q[7];
U1q(1.72892189988801*pi,0.9820230492367601*pi) q[8];
U1q(0.65166967608231*pi,0.219478838576312*pi) q[9];
U1q(3.273766869959622*pi,1.157889221073686*pi) q[10];
U1q(3.630163995588764*pi,1.832259623600023*pi) q[11];
U1q(0.744209268675153*pi,1.557261133985579*pi) q[12];
U1q(0.662996304595787*pi,0.664636695522398*pi) q[13];
U1q(0.672835531214822*pi,1.09365384492372*pi) q[14];
U1q(1.79404324808513*pi,0.62109671749298*pi) q[15];
U1q(0.600864576177457*pi,0.900830258346711*pi) q[16];
U1q(0.466228239347931*pi,1.883717465039091*pi) q[17];
U1q(1.70941482439998*pi,0.108806904827561*pi) q[18];
U1q(1.44240915390069*pi,1.4042393220805809*pi) q[19];
U1q(0.682308030034401*pi,0.681619733292032*pi) q[20];
U1q(0.214622063421408*pi,0.68148686762828*pi) q[21];
U1q(1.34133243734372*pi,0.81293654055808*pi) q[22];
U1q(0.705888054164057*pi,1.49886514009296*pi) q[23];
U1q(0.437784390973851*pi,0.276003660643797*pi) q[24];
U1q(1.53681179260693*pi,1.649417485966348*pi) q[25];
U1q(0.587144573740958*pi,1.804889967312042*pi) q[26];
U1q(0.502532998035601*pi,0.158896274518763*pi) q[27];
U1q(3.466913044426814*pi,0.390006512646173*pi) q[28];
U1q(0.285552412846801*pi,1.125408277184391*pi) q[29];
U1q(0.176397038859879*pi,1.8135388082329271*pi) q[30];
U1q(0.753968814579038*pi,0.86760582720265*pi) q[31];
U1q(0.594545144841839*pi,0.785250537910155*pi) q[32];
U1q(0.70760989923863*pi,1.15432530470915*pi) q[33];
U1q(0.667537222973326*pi,1.7832131839857621*pi) q[34];
U1q(1.64250307513009*pi,1.2266197039934421*pi) q[35];
U1q(0.64158737742627*pi,1.24648264793923*pi) q[36];
U1q(1.43487118062402*pi,0.100146069364511*pi) q[37];
U1q(0.565597489998154*pi,0.0861837932277214*pi) q[38];
U1q(1.44621422460151*pi,1.6738754201491859*pi) q[39];
RZZ(0.5*pi) q[36],q[0];
RZZ(0.5*pi) q[1],q[23];
RZZ(0.5*pi) q[2],q[28];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[5],q[35];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[29],q[11];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[22],q[37];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[32],q[38];
U1q(3.0303724819325852*pi,0.5647956398511819*pi) q[0];
U1q(1.52201297360748*pi,0.967081493272623*pi) q[1];
U1q(0.449173880085034*pi,1.880866672981873*pi) q[2];
U1q(1.6744694230609*pi,1.7625803233308672*pi) q[3];
U1q(1.52179997815772*pi,0.2666622217596739*pi) q[4];
U1q(1.7057573547071*pi,1.3483732807555626*pi) q[5];
U1q(3.727884196007781*pi,0.5492481307159434*pi) q[6];
U1q(1.91599711239576*pi,0.3535682932849905*pi) q[7];
U1q(1.79995501855172*pi,1.183356559688362*pi) q[8];
U1q(1.38797870374352*pi,0.33761909484437*pi) q[9];
U1q(3.804341615337225*pi,1.199845425618618*pi) q[10];
U1q(3.047348243251092*pi,0.9605355669394562*pi) q[11];
U1q(0.876177027319611*pi,0.91479146122926*pi) q[12];
U1q(1.52680870386591*pi,1.672532659408303*pi) q[13];
U1q(1.83462359713551*pi,1.93239377788764*pi) q[14];
U1q(3.772676525537464*pi,0.7326429936351122*pi) q[15];
U1q(1.57582392856201*pi,0.188682297489173*pi) q[16];
U1q(0.444792128460461*pi,0.5892750411967*pi) q[17];
U1q(3.612918765624249*pi,1.6868149408359638*pi) q[18];
U1q(1.51842675024759*pi,1.1371533332418524*pi) q[19];
U1q(1.69803197549911*pi,0.8553507483671401*pi) q[20];
U1q(0.402103152646397*pi,1.563212568613165*pi) q[21];
U1q(1.54453336376174*pi,0.9105482073369373*pi) q[22];
U1q(1.5182410649065*pi,0.017521357058519982*pi) q[23];
U1q(0.802054755671494*pi,1.6050707288766501*pi) q[24];
U1q(3.363779131050701*pi,1.9255593646762117*pi) q[25];
U1q(0.452900425279289*pi,0.038060334105489924*pi) q[26];
U1q(1.56306510642289*pi,1.1498413688457672*pi) q[27];
U1q(1.41336940239345*pi,1.9624115113422953*pi) q[28];
U1q(3.655470719075154*pi,1.4625971030331102*pi) q[29];
U1q(0.851159115900539*pi,1.574421026830618*pi) q[30];
U1q(1.6680298856099*pi,0.554915237942148*pi) q[31];
U1q(0.316455406357688*pi,1.8964399117843*pi) q[32];
U1q(0.832649304853407*pi,0.49550459424941007*pi) q[33];
U1q(1.43633856197406*pi,1.0030105975683998*pi) q[34];
U1q(3.15304665600731*pi,1.8191730350524185*pi) q[35];
U1q(0.210179033345704*pi,1.262472253841415*pi) q[36];
U1q(1.37386253937387*pi,1.0204927330211786*pi) q[37];
U1q(1.8391449516062*pi,1.7546263751300102*pi) q[38];
U1q(3.714724299498218*pi,0.13480721555109154*pi) q[39];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[8],q[20];
RZZ(0.5*pi) q[9],q[16];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[12],q[29];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[22],q[14];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[18],q[34];
RZZ(0.5*pi) q[36],q[21];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[37],q[30];
RZZ(0.5*pi) q[38],q[35];
U1q(1.93261925337103*pi,1.877204551321054*pi) q[0];
U1q(3.309950526153775*pi,0.5225816819097875*pi) q[1];
U1q(0.693987404346671*pi,0.88460998277476*pi) q[2];
U1q(0.308836580177496*pi,0.23575338865055906*pi) q[3];
U1q(1.24442599695042*pi,0.31569067108529714*pi) q[4];
U1q(0.222517438749087*pi,0.2678964357656697*pi) q[5];
U1q(1.72957817769491*pi,0.9042017206288053*pi) q[6];
U1q(1.49374272042561*pi,1.388850398274224*pi) q[7];
U1q(1.46296718089921*pi,1.2149946867126422*pi) q[8];
U1q(3.468690824554565*pi,1.9822822854137803*pi) q[9];
U1q(3.501488803209954*pi,1.5175727024414787*pi) q[10];
U1q(1.6200099114166*pi,0.2460545827243616*pi) q[11];
U1q(3.676196499648001*pi,1.3330043653082901*pi) q[12];
U1q(3.2477513725597342*pi,0.9247692940219799*pi) q[13];
U1q(3.259664065908132*pi,1.1991813520429888*pi) q[14];
U1q(3.2885926131671*pi,0.8469851573821172*pi) q[15];
U1q(3.302197852200708*pi,0.9564873950326718*pi) q[16];
U1q(1.77347402441269*pi,1.32699336031417*pi) q[17];
U1q(3.2466679968971057*pi,1.626167509345167*pi) q[18];
U1q(0.510009561014276*pi,1.3396766015452064*pi) q[19];
U1q(3.177800588277514*pi,0.7428815356495664*pi) q[20];
U1q(1.66664185987815*pi,0.35657262277159996*pi) q[21];
U1q(0.772962070571345*pi,1.827413296888567*pi) q[22];
U1q(3.224115063838133*pi,0.5656140435814478*pi) q[23];
U1q(1.33709557800662*pi,0.7290136232881901*pi) q[24];
U1q(3.234715051483668*pi,1.7767661949505325*pi) q[25];
U1q(0.629772056754758*pi,1.6867174479879599*pi) q[26];
U1q(1.86218027981521*pi,1.477157687817727*pi) q[27];
U1q(0.13331977095424*pi,1.9380266485063222*pi) q[28];
U1q(3.244744430509991*pi,0.914090804248926*pi) q[29];
U1q(0.773930984537618*pi,0.81799317417737*pi) q[30];
U1q(1.17053557276562*pi,0.04794791073421245*pi) q[31];
U1q(0.449068239847741*pi,1.57065208240491*pi) q[32];
U1q(3.310843237794809*pi,1.58183067113216*pi) q[33];
U1q(3.321973690245589*pi,1.0904172242884513*pi) q[34];
U1q(3.780791696414525*pi,0.40987184203051164*pi) q[35];
U1q(1.51872336046387*pi,0.4829925819490799*pi) q[36];
U1q(1.1736600419316*pi,1.7847091215980377*pi) q[37];
U1q(1.54004353839281*pi,0.3617461729539859*pi) q[38];
U1q(1.22901364135538*pi,1.1901275085357714*pi) q[39];
RZZ(0.5*pi) q[29],q[0];
RZZ(0.5*pi) q[1],q[9];
RZZ(0.5*pi) q[7],q[2];
RZZ(0.5*pi) q[31],q[3];
RZZ(0.5*pi) q[4],q[28];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[39],q[8];
RZZ(0.5*pi) q[10],q[24];
RZZ(0.5*pi) q[11],q[38];
RZZ(0.5*pi) q[27],q[12];
RZZ(0.5*pi) q[33],q[13];
RZZ(0.5*pi) q[14],q[30];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[22],q[18];
RZZ(0.5*pi) q[21],q[20];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[34],q[35];
U1q(1.22799856072324*pi,1.449653784077202*pi) q[0];
U1q(3.781234325543808*pi,0.9293812579269876*pi) q[1];
U1q(0.907796011091851*pi,0.3664571565051*pi) q[2];
U1q(0.728900156779565*pi,0.5717993829739672*pi) q[3];
U1q(3.466292652597994*pi,1.0381698795012255*pi) q[4];
U1q(0.281669239049081*pi,0.7130943512361996*pi) q[5];
U1q(0.14331455826719*pi,0.12739494522266548*pi) q[6];
U1q(1.76655090487726*pi,0.925865087190286*pi) q[7];
U1q(3.9318799070645976*pi,0.7481607828495229*pi) q[8];
U1q(1.7086615519103*pi,0.39684462741716064*pi) q[9];
U1q(3.625668203574004*pi,0.3549790356810485*pi) q[10];
U1q(1.47695804484561*pi,0.21134686444727258*pi) q[11];
U1q(1.37033910420192*pi,1.5294589213310053*pi) q[12];
U1q(3.834711574100774*pi,0.896808395617428*pi) q[13];
U1q(1.8337219858927*pi,0.1857937011054167*pi) q[14];
U1q(3.717039930847419*pi,0.205417288731216*pi) q[15];
U1q(1.60435155110745*pi,0.11210307550118226*pi) q[16];
U1q(1.44946672215852*pi,0.23009028014217447*pi) q[17];
U1q(1.05563332161429*pi,1.1320999990402738*pi) q[18];
U1q(0.395280488914693*pi,1.8350276221036212*pi) q[19];
U1q(1.20846370103728*pi,0.9392249312875065*pi) q[20];
U1q(3.6136029323904832*pi,1.3266301075882678*pi) q[21];
U1q(0.449690959345117*pi,1.954472102792578*pi) q[22];
U1q(1.36511301637336*pi,0.08420529650102093*pi) q[23];
U1q(3.615282367421413*pi,0.05388947914239717*pi) q[24];
U1q(1.27810644899875*pi,1.9419009275340435*pi) q[25];
U1q(0.580342074892259*pi,1.3499336358626701*pi) q[26];
U1q(0.605146933969195*pi,1.7507152839625402*pi) q[27];
U1q(0.662689168098749*pi,1.779446254823842*pi) q[28];
U1q(3.179075930010632*pi,0.7108067493148136*pi) q[29];
U1q(1.95761944695113*pi,0.5297811169678399*pi) q[30];
U1q(0.640715759919855*pi,1.7975417955957544*pi) q[31];
U1q(0.208015006506008*pi,1.1114829560108301*pi) q[32];
U1q(3.686893618117803*pi,1.6096318844507715*pi) q[33];
U1q(3.59110770496884*pi,0.4193155793158603*pi) q[34];
U1q(0.510381820631996*pi,1.2828351044615012*pi) q[35];
U1q(3.362749276579804*pi,0.6701410404805652*pi) q[36];
U1q(1.49979777664764*pi,0.9592223574874446*pi) q[37];
U1q(1.65193523586431*pi,1.065273966386326*pi) q[38];
U1q(3.501685776467149*pi,1.6196296233336822*pi) q[39];
RZZ(0.5*pi) q[24],q[0];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[5],q[4];
RZZ(0.5*pi) q[6],q[28];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[8],q[12];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[13],q[18];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[31],q[16];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[22],q[26];
RZZ(0.5*pi) q[37],q[29];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[33],q[36];
RZZ(0.5*pi) q[34],q[38];
U1q(1.3601787207639*pi,0.734809448537149*pi) q[0];
U1q(1.4747062554709*pi,0.0845150133010657*pi) q[1];
U1q(1.55716028856176*pi,1.7612607677926597*pi) q[2];
U1q(3.668544117368073*pi,0.9066501194300471*pi) q[3];
U1q(3.221687048173166*pi,0.26902441016293555*pi) q[4];
U1q(0.790661278870819*pi,0.019275412150279614*pi) q[5];
U1q(0.61636459779852*pi,0.037256493104025346*pi) q[6];
U1q(1.3435183304432*pi,0.6565415219834057*pi) q[7];
U1q(3.9838544033885124*pi,1.284395223752533*pi) q[8];
U1q(3.15836627244125*pi,1.979035792821529*pi) q[9];
U1q(3.299231818863815*pi,1.2390682969216087*pi) q[10];
U1q(1.11506212357089*pi,1.7430715359266569*pi) q[11];
U1q(1.48254518674593*pi,0.7851952450549851*pi) q[12];
U1q(1.37705792156214*pi,1.3851044493140927*pi) q[13];
U1q(1.62962307832512*pi,0.07436062502202656*pi) q[14];
U1q(3.344708543797033*pi,1.174428011950626*pi) q[15];
U1q(0.0521637690520478*pi,0.36925549833265237*pi) q[16];
U1q(0.217259347274404*pi,0.8339962620399843*pi) q[17];
U1q(1.32698217401382*pi,0.008275319349853927*pi) q[18];
U1q(1.13267237884665*pi,1.9867526435012914*pi) q[19];
U1q(3.193428226375299*pi,0.21369998088781417*pi) q[20];
U1q(0.750662005796271*pi,0.11322994536816777*pi) q[21];
U1q(0.522682320520628*pi,0.26439973328051725*pi) q[22];
U1q(1.01704114573269*pi,1.264374328532611*pi) q[23];
U1q(1.40153358669097*pi,0.6338214059501821*pi) q[24];
U1q(0.422384021066834*pi,1.7996464826630987*pi) q[25];
U1q(0.160383596655299*pi,1.2327758311296502*pi) q[26];
U1q(0.615429881366941*pi,0.89933176164124*pi) q[27];
U1q(0.779045707019585*pi,1.1524381313725325*pi) q[28];
U1q(1.57660158499144*pi,1.1790809462118634*pi) q[29];
U1q(1.3121980914182*pi,0.22682608607631272*pi) q[30];
U1q(0.0438364738164473*pi,0.11565938310043444*pi) q[31];
U1q(1.39639923800302*pi,0.28824208457253997*pi) q[32];
U1q(1.29439289106267*pi,1.0366732798614753*pi) q[33];
U1q(1.44674922188075*pi,0.9471689202444962*pi) q[34];
U1q(0.775742483980266*pi,0.7708224040514713*pi) q[35];
U1q(1.39665777052724*pi,1.759305168225799*pi) q[36];
U1q(1.57917435369576*pi,1.5211403500247442*pi) q[37];
U1q(1.77680729793983*pi,0.30184797432163446*pi) q[38];
U1q(3.490957264738758*pi,1.0890595414818014*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[2],q[18];
RZZ(0.5*pi) q[3],q[4];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[36],q[8];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[35];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[19],q[32];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[22],q[29];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[31],q[28];
RZZ(0.5*pi) q[30],q[34];
RZZ(0.5*pi) q[33],q[37];
U1q(0.638095185991976*pi,1.026814956325275*pi) q[0];
U1q(0.853965498149334*pi,0.7757149258720055*pi) q[1];
U1q(3.298021181983548*pi,0.49742988099888574*pi) q[2];
U1q(3.273205536066933*pi,1.906205223642793*pi) q[3];
U1q(1.62692403063618*pi,1.5761134266311094*pi) q[4];
U1q(1.57999011795454*pi,1.6681231455164998*pi) q[5];
U1q(1.35474793950032*pi,0.7115764726488649*pi) q[6];
U1q(1.65716560138696*pi,0.9196088194237779*pi) q[7];
U1q(3.445086590656371*pi,1.6523662390982627*pi) q[8];
U1q(1.76336289230725*pi,1.6208254200530092*pi) q[9];
U1q(3.3333101646886423*pi,1.693857479330351*pi) q[10];
U1q(3.822558323633744*pi,1.523142382569203*pi) q[11];
U1q(3.613187673572074*pi,0.3474053219839881*pi) q[12];
U1q(0.250368190342193*pi,0.037319815397892864*pi) q[13];
U1q(1.70092495297126*pi,1.9714862409648055*pi) q[14];
U1q(3.555540965375501*pi,0.46270647930035613*pi) q[15];
U1q(1.32313570399547*pi,0.06112710175828262*pi) q[16];
U1q(1.80609896604049*pi,0.694292419318205*pi) q[17];
U1q(3.28366204144737*pi,0.8500689960702879*pi) q[18];
U1q(3.325417928710061*pi,1.7422654781184326*pi) q[19];
U1q(3.590077446912295*pi,1.3216841763126137*pi) q[20];
U1q(1.85704486262193*pi,0.8269785894980473*pi) q[21];
U1q(0.785019158351967*pi,1.964341814063948*pi) q[22];
U1q(3.871369352743652*pi,1.039314231968212*pi) q[23];
U1q(3.184468273966276*pi,0.7779213729762615*pi) q[24];
U1q(0.486428723840699*pi,1.8566523423417185*pi) q[25];
U1q(0.807978646305*pi,0.8702361527158602*pi) q[26];
U1q(3.123258727824979*pi,0.022566160421209958*pi) q[27];
U1q(3.637650919824385*pi,1.0404927378000917*pi) q[28];
U1q(1.58719040118945*pi,1.789093870511243*pi) q[29];
U1q(1.65978735136442*pi,0.3734010350625727*pi) q[30];
U1q(3.323194449607033*pi,0.7380641525538145*pi) q[31];
U1q(3.589354401246562*pi,0.6376744411763156*pi) q[32];
U1q(1.721978627649*pi,0.1731108504585943*pi) q[33];
U1q(1.65780667262485*pi,0.5362568930872467*pi) q[34];
U1q(1.51034325054587*pi,1.843173125640381*pi) q[35];
U1q(0.131427253477005*pi,0.754343857524419*pi) q[36];
U1q(3.304428451042677*pi,1.8278029678385401*pi) q[37];
U1q(1.40075589484593*pi,1.477449405947194*pi) q[38];
U1q(3.681375318363423*pi,1.5539599136842464*pi) q[39];
RZZ(0.5*pi) q[27],q[0];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[6],q[30];
RZZ(0.5*pi) q[7],q[23];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[36],q[10];
RZZ(0.5*pi) q[12],q[11];
RZZ(0.5*pi) q[31],q[13];
RZZ(0.5*pi) q[14],q[16];
RZZ(0.5*pi) q[15],q[38];
RZZ(0.5*pi) q[17],q[29];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[37],q[25];
RZZ(0.5*pi) q[32],q[35];
U1q(0.420202278794686*pi,1.713165496856429*pi) q[0];
U1q(0.615580334196356*pi,1.211816421354766*pi) q[1];
U1q(1.55469802051734*pi,0.7379483214486129*pi) q[2];
U1q(1.54723801091872*pi,0.8933626017122638*pi) q[3];
U1q(0.507289802263416*pi,1.0160852596087295*pi) q[4];
U1q(1.30331626365632*pi,1.997714769394893*pi) q[5];
U1q(3.560842227205746*pi,0.981695324166596*pi) q[6];
U1q(0.964693722824144*pi,0.20071920341444827*pi) q[7];
U1q(1.12341773073989*pi,1.3811432528475338*pi) q[8];
U1q(1.61312821323923*pi,0.1700658956396719*pi) q[9];
U1q(0.668965387028013*pi,1.347295420314481*pi) q[10];
U1q(1.41325040665886*pi,1.2854737457998362*pi) q[11];
U1q(1.50938838135886*pi,1.7994724775981474*pi) q[12];
U1q(0.42045931350913*pi,0.3973331617806326*pi) q[13];
U1q(0.189667638283875*pi,1.1510485352123556*pi) q[14];
U1q(1.22168146740017*pi,1.8258005260710093*pi) q[15];
U1q(1.69870730083092*pi,0.07225118088418903*pi) q[16];
U1q(1.70934943245145*pi,1.819159760236487*pi) q[17];
U1q(1.63156191137606*pi,1.0194814904812892*pi) q[18];
U1q(1.30100841808077*pi,1.5273864545364662*pi) q[19];
U1q(1.40953027603767*pi,1.8683024080277506*pi) q[20];
U1q(1.52594187123634*pi,0.4042012297324957*pi) q[21];
U1q(0.451680479217191*pi,0.13172039998637786*pi) q[22];
U1q(0.752245139351417*pi,0.7492915822777215*pi) q[23];
U1q(1.70722992559334*pi,0.3801916725085821*pi) q[24];
U1q(0.193892436315279*pi,1.8922647988862986*pi) q[25];
U1q(0.604486978267491*pi,0.8706247729665499*pi) q[26];
U1q(1.63403106976407*pi,0.0290163535484238*pi) q[27];
U1q(1.31855719148014*pi,1.680132374587239*pi) q[28];
U1q(0.296733593975856*pi,0.8256537439885427*pi) q[29];
U1q(3.756688068300818*pi,1.1416672415779905*pi) q[30];
U1q(3.164503262363411*pi,0.6622122781249749*pi) q[31];
U1q(1.6468315159498*pi,0.11779085026916558*pi) q[32];
U1q(1.734580507483*pi,0.4367455911641488*pi) q[33];
U1q(1.38315218724198*pi,1.4523835209454568*pi) q[34];
U1q(1.49260062446413*pi,1.9198047079676233*pi) q[35];
U1q(0.651304761708173*pi,1.0231733842110184*pi) q[36];
U1q(0.760542045136463*pi,0.2807756755591404*pi) q[37];
U1q(1.54810826670686*pi,1.2775748528941069*pi) q[38];
U1q(0.30469433926821*pi,0.15192512615208553*pi) q[39];
rz(2.286834503143571*pi) q[0];
rz(2.788183578645234*pi) q[1];
rz(3.262051678551387*pi) q[2];
rz(1.1066373982877362*pi) q[3];
rz(0.9839147403912705*pi) q[4];
rz(0.0022852306051071025*pi) q[5];
rz(1.018304675833404*pi) q[6];
rz(1.7992807965855517*pi) q[7];
rz(0.6188567471524662*pi) q[8];
rz(3.829934104360328*pi) q[9];
rz(2.652704579685519*pi) q[10];
rz(0.7145262542001638*pi) q[11];
rz(2.2005275224018526*pi) q[12];
rz(3.6026668382193674*pi) q[13];
rz(2.8489514647876444*pi) q[14];
rz(0.17419947392899068*pi) q[15];
rz(1.927748819115811*pi) q[16];
rz(0.18084023976351293*pi) q[17];
rz(2.980518509518711*pi) q[18];
rz(2.472613545463534*pi) q[19];
rz(2.1316975919722494*pi) q[20];
rz(1.5957987702675043*pi) q[21];
rz(3.868279600013622*pi) q[22];
rz(3.2507084177222785*pi) q[23];
rz(1.619808327491418*pi) q[24];
rz(0.10773520111370138*pi) q[25];
rz(1.12937522703345*pi) q[26];
rz(3.970983646451576*pi) q[27];
rz(2.319867625412761*pi) q[28];
rz(3.1743462560114573*pi) q[29];
rz(0.8583327584220095*pi) q[30];
rz(1.3377877218750251*pi) q[31];
rz(1.8822091497308344*pi) q[32];
rz(3.563254408835851*pi) q[33];
rz(0.5476164790545432*pi) q[34];
rz(0.08019529203237674*pi) q[35];
rz(0.9768266157889816*pi) q[36];
rz(3.7192243244408596*pi) q[37];
rz(2.722425147105893*pi) q[38];
rz(3.8480748738479145*pi) q[39];
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
