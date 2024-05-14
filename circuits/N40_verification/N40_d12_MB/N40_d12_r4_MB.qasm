OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.284768404436434*pi,0.634297658185948*pi) q[0];
U1q(0.177612618811872*pi,0.161920851547004*pi) q[1];
U1q(1.52176136924676*pi,1.5875733766832871*pi) q[2];
U1q(1.32477060936149*pi,0.9423321262222786*pi) q[3];
U1q(3.396445174238741*pi,1.3772576164534782*pi) q[4];
U1q(0.76022306291728*pi,1.92295556044439*pi) q[5];
U1q(0.715232531077296*pi,1.9710300938714722*pi) q[6];
U1q(1.51215961605759*pi,1.5707945207077783*pi) q[7];
U1q(0.749785275321302*pi,1.27978930989962*pi) q[8];
U1q(0.73354701194431*pi,1.0851129259640682*pi) q[9];
U1q(0.157056753701973*pi,1.634570901820126*pi) q[10];
U1q(0.585698561120543*pi,0.232101929289892*pi) q[11];
U1q(0.276751529625024*pi,0.819787405919218*pi) q[12];
U1q(0.383530130030445*pi,0.0279512113995286*pi) q[13];
U1q(1.55284706236463*pi,0.03696765832030316*pi) q[14];
U1q(0.91258957959159*pi,1.7984280290539*pi) q[15];
U1q(1.70943963520104*pi,1.8860251291828096*pi) q[16];
U1q(3.555004046078506*pi,1.0062756604663385*pi) q[17];
U1q(0.428670539148318*pi,1.8754041505926131*pi) q[18];
U1q(0.340186850576616*pi,0.99714500279187*pi) q[19];
U1q(0.757882507629551*pi,0.938045055479362*pi) q[20];
U1q(0.782452364586208*pi,1.881188453988237*pi) q[21];
U1q(0.412913839845259*pi,0.846049995217075*pi) q[22];
U1q(0.133365104753342*pi,1.22072434697692*pi) q[23];
U1q(1.56294963102449*pi,1.1255938178284595*pi) q[24];
U1q(0.678730956324851*pi,1.765268372553078*pi) q[25];
U1q(1.559657612391*pi,0.40273320632947474*pi) q[26];
U1q(0.539729310596461*pi,1.101119805104907*pi) q[27];
U1q(3.588193532354429*pi,0.9827531973941017*pi) q[28];
U1q(0.809354019819846*pi,0.23529078842059*pi) q[29];
U1q(1.27675961794917*pi,1.3946829266304188*pi) q[30];
U1q(1.78703550405397*pi,0.0967002434388886*pi) q[31];
U1q(0.589694067604941*pi,0.964211784936073*pi) q[32];
U1q(1.59766124961709*pi,1.72354000406956*pi) q[33];
U1q(1.27669597581965*pi,1.4865217828622468*pi) q[34];
U1q(1.56546580232964*pi,0.46276778082328063*pi) q[35];
U1q(1.53602901629055*pi,0.602896085331113*pi) q[36];
U1q(0.791080946318617*pi,0.808350850125405*pi) q[37];
U1q(1.47063126929316*pi,1.8843157917097446*pi) q[38];
U1q(1.6592556120199*pi,0.3995983269640686*pi) q[39];
RZZ(0.5*pi) q[0],q[32];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[20],q[5];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[8],q[13];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[27],q[10];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[30],q[12];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[15],q[34];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[39],q[21];
RZZ(0.5*pi) q[38],q[25];
RZZ(0.5*pi) q[31],q[28];
RZZ(0.5*pi) q[36],q[37];
U1q(0.565584274931472*pi,0.46244675914879996*pi) q[0];
U1q(0.533611246473373*pi,0.0909064329017002*pi) q[1];
U1q(0.823093774526977*pi,1.5388770827086273*pi) q[2];
U1q(0.302583496012672*pi,0.17774411819750757*pi) q[3];
U1q(0.71717217637225*pi,1.580365428301218*pi) q[4];
U1q(0.64557993593859*pi,1.76938643243781*pi) q[5];
U1q(0.81578563336093*pi,0.0019439846645399328*pi) q[6];
U1q(0.676112744726222*pi,0.7135675303179463*pi) q[7];
U1q(0.940783452942816*pi,1.843053935072664*pi) q[8];
U1q(0.437715432825031*pi,1.3666209228106796*pi) q[9];
U1q(0.309756828409122*pi,0.7479449412516201*pi) q[10];
U1q(0.290560887431452*pi,0.6153896159380801*pi) q[11];
U1q(0.595898350306838*pi,0.47839070377327*pi) q[12];
U1q(0.327107149101463*pi,0.15946444813040994*pi) q[13];
U1q(0.453160530162942*pi,0.7021720545123435*pi) q[14];
U1q(0.220745913417157*pi,0.17838369770905005*pi) q[15];
U1q(0.1972847372104*pi,0.00911672266874941*pi) q[16];
U1q(0.560945493017913*pi,0.9745538000723486*pi) q[17];
U1q(0.884219453659948*pi,0.5762899176763598*pi) q[18];
U1q(0.161096008078001*pi,0.8762405454355902*pi) q[19];
U1q(0.799809319061094*pi,0.411546119700686*pi) q[20];
U1q(0.562841039891771*pi,0.6905373598273399*pi) q[21];
U1q(0.628158238861243*pi,0.09985266709659002*pi) q[22];
U1q(0.125520738099871*pi,0.8707391339630899*pi) q[23];
U1q(0.768436399800032*pi,0.8003423196141592*pi) q[24];
U1q(0.685818491460811*pi,0.4439850500825102*pi) q[25];
U1q(0.680728761593106*pi,0.45418001076325476*pi) q[26];
U1q(0.0767048656117201*pi,1.89960619403855*pi) q[27];
U1q(0.447986352946799*pi,0.6976072865789847*pi) q[28];
U1q(0.503181208673684*pi,0.36524493269832004*pi) q[29];
U1q(0.0965640710451754*pi,0.45235292680131867*pi) q[30];
U1q(0.264280520823768*pi,0.8434140630085483*pi) q[31];
U1q(0.253875369555045*pi,0.77609810070006*pi) q[32];
U1q(0.56819576598558*pi,0.71238395063114*pi) q[33];
U1q(0.616293141844913*pi,1.9700079197198264*pi) q[34];
U1q(0.498309037622699*pi,1.2642899849411005*pi) q[35];
U1q(0.296656206921941*pi,1.128963558445153*pi) q[36];
U1q(0.62422525952732*pi,0.5539856917392001*pi) q[37];
U1q(0.0279513113714409*pi,0.12646893047368457*pi) q[38];
U1q(0.401542089105375*pi,0.48731816341063894*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[1],q[17];
RZZ(0.5*pi) q[12],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[7],q[6];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[39],q[10];
RZZ(0.5*pi) q[35],q[11];
RZZ(0.5*pi) q[38],q[13];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[31],q[20];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[26],q[33];
RZZ(0.5*pi) q[27],q[37];
U1q(0.388781857859265*pi,0.28520825236719016*pi) q[0];
U1q(0.8198315015968*pi,1.5021883534401397*pi) q[1];
U1q(0.616302594596266*pi,1.2587082207545368*pi) q[2];
U1q(0.803799036979463*pi,0.35390587298778864*pi) q[3];
U1q(0.163130407625664*pi,1.5280642216511984*pi) q[4];
U1q(0.600404136456905*pi,1.76230223844652*pi) q[5];
U1q(0.534844139740851*pi,1.40985739511708*pi) q[6];
U1q(0.800489799346925*pi,0.7820874488976086*pi) q[7];
U1q(0.944092836148875*pi,0.7879529050121501*pi) q[8];
U1q(0.633727500499019*pi,1.9808062114380904*pi) q[9];
U1q(0.734092058430003*pi,0.7020968176739002*pi) q[10];
U1q(0.442303056286245*pi,1.1626240573655098*pi) q[11];
U1q(0.59123671288135*pi,1.0039884259099896*pi) q[12];
U1q(0.884062505275403*pi,0.15525047063615993*pi) q[13];
U1q(0.578619785980378*pi,1.1214180482547027*pi) q[14];
U1q(0.640387843921952*pi,1.2481266814833498*pi) q[15];
U1q(0.505274273539633*pi,1.6512106659632098*pi) q[16];
U1q(0.282085280080274*pi,0.1887945381824494*pi) q[17];
U1q(0.70102276081333*pi,1.0298297951062798*pi) q[18];
U1q(0.206121721125056*pi,0.6030989617889801*pi) q[19];
U1q(0.531653062334403*pi,0.68336997477275*pi) q[20];
U1q(0.451197904780388*pi,0.74771621945911*pi) q[21];
U1q(0.438796644005431*pi,1.9222503746338502*pi) q[22];
U1q(0.657976658093836*pi,1.6001725070615702*pi) q[23];
U1q(0.830528619964291*pi,1.22399047354042*pi) q[24];
U1q(0.472941551538038*pi,0.3517934388439503*pi) q[25];
U1q(0.778343509509143*pi,0.6259477709610648*pi) q[26];
U1q(0.546846691595769*pi,0.6766929168003397*pi) q[27];
U1q(0.459946810999815*pi,0.4575811532181615*pi) q[28];
U1q(0.284363806712946*pi,0.59689583340875*pi) q[29];
U1q(0.727420487609067*pi,1.961231109267489*pi) q[30];
U1q(0.579400639711193*pi,0.6007117373541488*pi) q[31];
U1q(0.573483576430486*pi,1.4298141548186099*pi) q[32];
U1q(0.971616273908412*pi,1.4749456206208302*pi) q[33];
U1q(0.243501141705711*pi,0.9531687232131372*pi) q[34];
U1q(0.901419825678785*pi,1.830257343238591*pi) q[35];
U1q(0.820597456415735*pi,1.8738456141410529*pi) q[36];
U1q(0.121377667037663*pi,0.45274233106508*pi) q[37];
U1q(0.568149189557907*pi,0.885615008147945*pi) q[38];
U1q(0.892464735784512*pi,0.8798610750648388*pi) q[39];
RZZ(0.5*pi) q[0],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[31],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[16],q[11];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[38],q[14];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[20],q[25];
RZZ(0.5*pi) q[24],q[21];
RZZ(0.5*pi) q[27],q[22];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[28],q[36];
RZZ(0.5*pi) q[30],q[37];
RZZ(0.5*pi) q[39],q[32];
U1q(0.744639723209807*pi,0.9238279764971304*pi) q[0];
U1q(0.643023012068782*pi,1.7785493367709204*pi) q[1];
U1q(0.152665204562775*pi,0.420653145343727*pi) q[2];
U1q(0.302241658268885*pi,1.4315131843817381*pi) q[3];
U1q(0.456709683883986*pi,1.3288302123429983*pi) q[4];
U1q(0.208491893601783*pi,1.4306019243171901*pi) q[5];
U1q(0.41210352369922*pi,1.3535486286796008*pi) q[6];
U1q(0.707630709234578*pi,0.0940816257838284*pi) q[7];
U1q(0.765316366624849*pi,0.6566273178260502*pi) q[8];
U1q(0.601372588725213*pi,1.78533837434121*pi) q[9];
U1q(0.588152101591854*pi,1.1076211975370995*pi) q[10];
U1q(0.326714432578041*pi,1.2938548503009804*pi) q[11];
U1q(0.611666926356392*pi,0.43918077811741973*pi) q[12];
U1q(0.17282220721696*pi,1.7895125492102997*pi) q[13];
U1q(0.447978825621065*pi,0.09486200032838266*pi) q[14];
U1q(0.541359392853087*pi,1.1832505246515197*pi) q[15];
U1q(0.553067556436338*pi,0.17561984523172924*pi) q[16];
U1q(0.293183715594288*pi,1.0634853696253384*pi) q[17];
U1q(0.856584670723727*pi,1.82807068444842*pi) q[18];
U1q(0.769441866165592*pi,0.21869804609587984*pi) q[19];
U1q(0.765223254755653*pi,0.5952140633555203*pi) q[20];
U1q(0.168777176986666*pi,0.6045071718173602*pi) q[21];
U1q(0.286959527148715*pi,1.6673281832416604*pi) q[22];
U1q(0.143050501579427*pi,0.42701173598154973*pi) q[23];
U1q(0.215689103448845*pi,1.2207510005207798*pi) q[24];
U1q(0.876326028192074*pi,0.9019320213891806*pi) q[25];
U1q(0.539028126932466*pi,0.23459689607006506*pi) q[26];
U1q(0.88448422986685*pi,0.8721500542146998*pi) q[27];
U1q(0.791349433503897*pi,0.6136071823638618*pi) q[28];
U1q(0.906421843080476*pi,0.14186613105855006*pi) q[29];
U1q(0.820989726247516*pi,0.7624919158868586*pi) q[30];
U1q(0.747799051200784*pi,0.3424513040521484*pi) q[31];
U1q(0.203357512034413*pi,0.09196945784499988*pi) q[32];
U1q(0.585721492516096*pi,1.0414070884869702*pi) q[33];
U1q(0.533847643803775*pi,1.4959521467454273*pi) q[34];
U1q(0.824458792672174*pi,0.5000888035444611*pi) q[35];
U1q(0.328594092039002*pi,0.6841662899806229*pi) q[36];
U1q(0.835234043521264*pi,1.6646702190276397*pi) q[37];
U1q(0.517120564713064*pi,0.3978288786178048*pi) q[38];
U1q(0.0678612087504365*pi,0.9854496757011182*pi) q[39];
RZZ(0.5*pi) q[0],q[24];
RZZ(0.5*pi) q[16],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[21];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[7],q[39];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[12],q[9];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[36],q[13];
RZZ(0.5*pi) q[31],q[14];
RZZ(0.5*pi) q[23],q[17];
RZZ(0.5*pi) q[27],q[19];
RZZ(0.5*pi) q[20],q[33];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[32],q[28];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[38],q[37];
U1q(0.839340369522799*pi,0.7299763408964406*pi) q[0];
U1q(0.645057520859889*pi,0.10655820566242014*pi) q[1];
U1q(0.691303630057864*pi,1.6629601654172674*pi) q[2];
U1q(0.376276393840017*pi,0.1654230815171882*pi) q[3];
U1q(0.885916498305303*pi,1.402629008004709*pi) q[4];
U1q(0.304796685845117*pi,1.6644942363479904*pi) q[5];
U1q(0.599506715332559*pi,1.2161408637538997*pi) q[6];
U1q(0.362315452534769*pi,1.8122109103843584*pi) q[7];
U1q(0.21338934268116*pi,1.7320855763846197*pi) q[8];
U1q(0.721908756839425*pi,1.1984960585328004*pi) q[9];
U1q(0.608859749362176*pi,0.6355078939993994*pi) q[10];
U1q(0.617011391108017*pi,0.3138769131564896*pi) q[11];
U1q(0.570967215139066*pi,1.2458870258221992*pi) q[12];
U1q(0.437132182754675*pi,1.8082399364970794*pi) q[13];
U1q(0.72486712172394*pi,0.18309482907262442*pi) q[14];
U1q(0.62539689095624*pi,1.817685521164*pi) q[15];
U1q(0.678530227738261*pi,1.24722836105318*pi) q[16];
U1q(0.67846389156235*pi,1.6081338649271775*pi) q[17];
U1q(0.162567441658159*pi,1.3435519102858997*pi) q[18];
U1q(0.89269653453426*pi,0.6383015569166002*pi) q[19];
U1q(0.412125683238483*pi,0.013304363303079825*pi) q[20];
U1q(0.690849173415118*pi,0.20786629391479927*pi) q[21];
U1q(0.727737707795885*pi,0.5669562326525499*pi) q[22];
U1q(0.277495464844106*pi,1.0850730747041393*pi) q[23];
U1q(0.20250380152745*pi,1.68426890784243*pi) q[24];
U1q(0.723665678904415*pi,1.3104179821181*pi) q[25];
U1q(0.459457234478127*pi,1.3462025744314534*pi) q[26];
U1q(0.696509977854147*pi,0.4949139954910997*pi) q[27];
U1q(0.775937841757167*pi,1.7702788067722715*pi) q[28];
U1q(0.375363249265793*pi,0.4788550668225504*pi) q[29];
U1q(0.492865996236306*pi,1.4413591204946385*pi) q[30];
U1q(0.319789642132472*pi,1.9465777509645896*pi) q[31];
U1q(0.173171438410749*pi,0.8939887453156503*pi) q[32];
U1q(0.705099418553623*pi,0.2743569634158707*pi) q[33];
U1q(0.75309229601627*pi,0.3918822872134484*pi) q[34];
U1q(0.347025961785994*pi,1.9093607006729414*pi) q[35];
U1q(0.452770803769749*pi,0.07299843160637298*pi) q[36];
U1q(0.491225778119674*pi,0.5753328454606805*pi) q[37];
U1q(0.16872514915377*pi,0.3645550882581432*pi) q[38];
U1q(0.753516102127152*pi,0.6412043623560688*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[2],q[13];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[6],q[21];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[14],q[9];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[12],q[22];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[32],q[17];
RZZ(0.5*pi) q[26],q[18];
RZZ(0.5*pi) q[28],q[20];
RZZ(0.5*pi) q[30],q[25];
RZZ(0.5*pi) q[27],q[34];
RZZ(0.5*pi) q[29],q[33];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[35],q[36];
U1q(0.557654480091414*pi,1.9246030484769001*pi) q[0];
U1q(0.700260535049162*pi,0.7648286971367*pi) q[1];
U1q(0.488079819297742*pi,0.007938691513775353*pi) q[2];
U1q(0.227560317592966*pi,0.7023823877796787*pi) q[3];
U1q(0.281037738036616*pi,0.4235366150731785*pi) q[4];
U1q(0.82422239654206*pi,0.11905951685603*pi) q[5];
U1q(0.728426667231722*pi,0.8179548771379004*pi) q[6];
U1q(0.246857852214771*pi,1.8578837092391982*pi) q[7];
U1q(0.443529475670051*pi,0.3580799407883397*pi) q[8];
U1q(0.547714862897013*pi,1.4955838951174005*pi) q[9];
U1q(0.831901214657016*pi,0.13278996900129947*pi) q[10];
U1q(0.809942182321394*pi,0.9697723526514004*pi) q[11];
U1q(0.646247914645142*pi,1.7724623089142*pi) q[12];
U1q(0.665241403600323*pi,1.8368791707764007*pi) q[13];
U1q(0.682813870524349*pi,1.687940925525604*pi) q[14];
U1q(0.550158906157991*pi,1.5126420408846997*pi) q[15];
U1q(0.694319211629141*pi,0.12355890901284994*pi) q[16];
U1q(0.700534039448028*pi,0.40118032983743745*pi) q[17];
U1q(0.212204154815017*pi,0.020613349975999284*pi) q[18];
U1q(0.3350814984495*pi,0.3463753495571993*pi) q[19];
U1q(0.564524478705747*pi,0.7872307382474109*pi) q[20];
U1q(0.321028797892439*pi,1.0982473486205002*pi) q[21];
U1q(0.776841447894585*pi,1.2339295019349308*pi) q[22];
U1q(0.449320027294383*pi,0.4252800527826004*pi) q[23];
U1q(0.382046560461909*pi,0.15060136368045995*pi) q[24];
U1q(0.79202225880366*pi,1.0991511439867008*pi) q[25];
U1q(0.115703635348385*pi,0.5977382309602142*pi) q[26];
U1q(0.307550064918846*pi,0.006593950143200189*pi) q[27];
U1q(0.630795281436044*pi,1.9370999216002023*pi) q[28];
U1q(0.399909184852221*pi,0.5639832860910996*pi) q[29];
U1q(0.722481083320559*pi,0.08618407630401848*pi) q[30];
U1q(0.810786784725172*pi,1.0893433503948895*pi) q[31];
U1q(0.533617782745721*pi,1.3560255402342705*pi) q[32];
U1q(0.288055551098763*pi,0.5737040517922605*pi) q[33];
U1q(0.394825584497877*pi,0.8802416998590452*pi) q[34];
U1q(0.822218212430462*pi,0.984188850114581*pi) q[35];
U1q(0.322091634651958*pi,0.18175805887701202*pi) q[36];
U1q(0.580268028632434*pi,1.0296156663975005*pi) q[37];
U1q(0.236357913020531*pi,0.9432524996345446*pi) q[38];
U1q(0.156619932202308*pi,1.0779983209053885*pi) q[39];
RZZ(0.5*pi) q[0],q[21];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[4],q[3];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[29],q[7];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[32],q[11];
RZZ(0.5*pi) q[20],q[13];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[18],q[19];
RZZ(0.5*pi) q[25],q[22];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[28],q[37];
RZZ(0.5*pi) q[38],q[36];
U1q(0.214439891598629*pi,0.8705029477909996*pi) q[0];
U1q(0.726656136112595*pi,1.1100751735204994*pi) q[1];
U1q(0.507660316933054*pi,1.8999421043206866*pi) q[2];
U1q(0.759113943503839*pi,1.6920574639847796*pi) q[3];
U1q(0.727953524363071*pi,0.6230151250042795*pi) q[4];
U1q(0.211788909039742*pi,1.8366496417109008*pi) q[5];
U1q(0.894793700856223*pi,1.9250269357624*pi) q[6];
U1q(0.573170386570246*pi,0.6088929800907792*pi) q[7];
U1q(0.766532488918126*pi,1.1437084519916993*pi) q[8];
U1q(0.638714740928563*pi,1.1497108883396*pi) q[9];
U1q(0.552784238207873*pi,0.7874941808935993*pi) q[10];
U1q(0.423868472816842*pi,1.1685759343695992*pi) q[11];
U1q(0.279601336113047*pi,0.9107867982355984*pi) q[12];
U1q(0.300151384141506*pi,0.13460194602519948*pi) q[13];
U1q(0.445961301609501*pi,1.3143016774504055*pi) q[14];
U1q(0.238203079274885*pi,0.8879488953391004*pi) q[15];
U1q(0.700737629655454*pi,0.2408330843827109*pi) q[16];
U1q(0.569591605243784*pi,1.0403242203648375*pi) q[17];
U1q(0.515746210017001*pi,1.156368509827999*pi) q[18];
U1q(0.592919493221392*pi,0.7500576174864015*pi) q[19];
U1q(0.273005138597477*pi,1.0062483256483006*pi) q[20];
U1q(0.831719431270745*pi,0.2721017668407999*pi) q[21];
U1q(0.424383006217435*pi,0.04820821092735983*pi) q[22];
U1q(0.359116315574106*pi,1.8102247571949004*pi) q[23];
U1q(0.402263450758736*pi,0.8955888137656594*pi) q[24];
U1q(0.594674761561237*pi,0.9260871502381995*pi) q[25];
U1q(0.594487401645333*pi,1.370779765177474*pi) q[26];
U1q(0.614026633026596*pi,0.2549357738870004*pi) q[27];
U1q(0.141253124216132*pi,0.031894870598701885*pi) q[28];
U1q(0.207393195950351*pi,0.9979394916333*pi) q[29];
U1q(0.413230833426997*pi,1.9723915640466174*pi) q[30];
U1q(0.255699620814435*pi,1.512246744021688*pi) q[31];
U1q(0.317775968822366*pi,1.5240954158565998*pi) q[32];
U1q(0.227252572072074*pi,1.7904835359187103*pi) q[33];
U1q(0.604949325298451*pi,1.8957598715418484*pi) q[34];
U1q(0.592922954815946*pi,0.7759200942293809*pi) q[35];
U1q(0.652804121098344*pi,1.4215861615560126*pi) q[36];
U1q(0.480470230132322*pi,0.15814408904640054*pi) q[37];
U1q(0.572673631903807*pi,0.2634833144764439*pi) q[38];
U1q(0.561847716237641*pi,0.027020479079869375*pi) q[39];
rz(1.3101685029366*pi) q[0];
rz(2.3525929873579994*pi) q[1];
rz(0.30274526635211174*pi) q[2];
rz(1.7198822590094203*pi) q[3];
rz(0.9687312965285209*pi) q[4];
rz(3.6397955139255007*pi) q[5];
rz(0.35238947281960087*pi) q[6];
rz(1.698596500331222*pi) q[7];
rz(2.7983243746677005*pi) q[8];
rz(3.5570459338325993*pi) q[9];
rz(1.7009309520865017*pi) q[10];
rz(1.3296117558214*pi) q[11];
rz(0.42237352559769903*pi) q[12];
rz(2.158670675163499*pi) q[13];
rz(0.9855836685854946*pi) q[14];
rz(1.7913401887877*pi) q[15];
rz(0.8416034696637897*pi) q[16];
rz(2.0040039618861627*pi) q[17];
rz(0.8181417720638997*pi) q[18];
rz(2.0160351973029016*pi) q[19];
rz(0.6237571989219006*pi) q[20];
rz(3.8971995943668*pi) q[21];
rz(2.402706299619499*pi) q[22];
rz(2.7822756780508993*pi) q[23];
rz(2.9292480486733403*pi) q[24];
rz(3.103966449911301*pi) q[25];
rz(2.222893104504527*pi) q[26];
rz(1.1886856137503017*pi) q[27];
rz(3.673241533222699*pi) q[28];
rz(1.8454207206392006*pi) q[29];
rz(0.7338879406075822*pi) q[30];
rz(0.8816934717429099*pi) q[31];
rz(2.705526110190201*pi) q[32];
rz(3.467250532215939*pi) q[33];
rz(1.3950688723970543*pi) q[34];
rz(0.03776362209761963*pi) q[35];
rz(3.675023873739386*pi) q[36];
rz(1.6016016608311006*pi) q[37];
rz(3.6877620022183564*pi) q[38];
rz(1.05705830315903*pi) q[39];
U1q(1.21443989159863*pi,1.180671450727556*pi) q[0];
U1q(1.7266561361126*pi,0.462668160878428*pi) q[1];
U1q(1.50766031693305*pi,1.20268737067288*pi) q[2];
U1q(1.75911394350384*pi,0.41193972299421*pi) q[3];
U1q(1.72795352436307*pi,0.591746421532837*pi) q[4];
U1q(0.211788909039742*pi,0.476445155636317*pi) q[5];
U1q(1.89479370085622*pi,1.27741640858203*pi) q[6];
U1q(0.573170386570246*pi,1.3074894804219461*pi) q[7];
U1q(1.76653248891813*pi,0.942032826659388*pi) q[8];
U1q(1.63871474092856*pi,1.7067568221722231*pi) q[9];
U1q(0.552784238207873*pi,1.48842513298015*pi) q[10];
U1q(1.42386847281684*pi,1.498187690190948*pi) q[11];
U1q(0.279601336113047*pi,0.333160323833287*pi) q[12];
U1q(0.300151384141506*pi,1.293272621188654*pi) q[13];
U1q(1.4459613016095*pi,1.299885346035937*pi) q[14];
U1q(0.238203079274885*pi,1.679289084126748*pi) q[15];
U1q(0.700737629655454*pi,0.0824365540465242*pi) q[16];
U1q(1.56959160524378*pi,0.0443281822510319*pi) q[17];
U1q(0.515746210017001*pi,0.974510281891978*pi) q[18];
U1q(0.592919493221392*pi,1.766092814789373*pi) q[19];
U1q(0.273005138597477*pi,0.630005524570151*pi) q[20];
U1q(0.831719431270745*pi,1.16930136120762*pi) q[21];
U1q(3.424383006217435*pi,1.450914510546834*pi) q[22];
U1q(0.359116315574106*pi,1.592500435245813*pi) q[23];
U1q(0.402263450758736*pi,0.824836862439006*pi) q[24];
U1q(0.594674761561237*pi,1.030053600149479*pi) q[25];
U1q(0.594487401645333*pi,0.593672869682065*pi) q[26];
U1q(0.614026633026596*pi,0.443621387637299*pi) q[27];
U1q(1.14125312421613*pi,0.70513640382139*pi) q[28];
U1q(0.207393195950351*pi,1.84336021227254*pi) q[29];
U1q(3.413230833426997*pi,1.7062795046541739*pi) q[30];
U1q(1.25569962081444*pi,1.3939402157645469*pi) q[31];
U1q(0.317775968822366*pi,1.2296215260467491*pi) q[32];
U1q(0.227252572072074*pi,0.257734068134618*pi) q[33];
U1q(0.604949325298451*pi,0.290828743938855*pi) q[34];
U1q(3.592922954815946*pi,1.813683716326971*pi) q[35];
U1q(0.652804121098344*pi,0.0966100352954249*pi) q[36];
U1q(1.48047023013232*pi,0.75974574987745*pi) q[37];
U1q(0.572673631903807*pi,0.9512453166947801*pi) q[38];
U1q(3.561847716237641*pi,0.0840787822388953*pi) q[39];
RZZ(0.5*pi) q[0],q[21];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[4],q[3];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[29],q[7];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[32],q[11];
RZZ(0.5*pi) q[20],q[13];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[18],q[19];
RZZ(0.5*pi) q[25],q[22];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[28],q[37];
RZZ(0.5*pi) q[38],q[36];
U1q(3.557654480091414*pi,0.12657135004164277*pi) q[0];
U1q(1.70026053504916*pi,0.8079146372621491*pi) q[1];
U1q(1.48807981929774*pi,1.0946907834798232*pi) q[2];
U1q(3.772439682407034*pi,1.4016147991993595*pi) q[3];
U1q(1.28103773803662*pi,1.7912249314639135*pi) q[4];
U1q(3.82422239654206*pi,1.758855030781492*pi) q[5];
U1q(3.271573332768278*pi,1.3844884672065694*pi) q[6];
U1q(1.24685785221477*pi,1.5564802095703998*pi) q[7];
U1q(3.556470524329949*pi,1.7276613378627645*pi) q[8];
U1q(1.54771486289701*pi,0.36088381539440695*pi) q[9];
U1q(0.831901214657016*pi,0.83372092108788*pi) q[10];
U1q(3.190057817678606*pi,0.6969912719091533*pi) q[11];
U1q(0.646247914645142*pi,0.19483583451185993*pi) q[12];
U1q(1.66524140360032*pi,1.9955498459399301*pi) q[13];
U1q(1.68281387052435*pi,0.9262460979607696*pi) q[14];
U1q(0.550158906157991*pi,0.30398222967236*pi) q[15];
U1q(1.69431921162914*pi,1.9651623786766201*pi) q[16];
U1q(1.70053403944803*pi,1.6834720727784673*pi) q[17];
U1q(1.21220415481502*pi,0.838755122039924*pi) q[18];
U1q(0.3350814984495*pi,0.36241054686017016*pi) q[19];
U1q(1.56452447870575*pi,1.410987937169295*pi) q[20];
U1q(0.321028797892439*pi,1.9954469429873*pi) q[21];
U1q(1.77684144789459*pi,0.2651932195392611*pi) q[22];
U1q(0.449320027294383*pi,1.2075557308335299*pi) q[23];
U1q(0.382046560461909*pi,0.0798494123537857*pi) q[24];
U1q(0.79202225880366*pi,1.203117593897997*pi) q[25];
U1q(0.115703635348385*pi,0.8206313354647901*pi) q[26];
U1q(0.307550064918846*pi,0.19527956389344991*pi) q[27];
U1q(3.369204718563955*pi,0.7999313528198976*pi) q[28];
U1q(1.39990918485222*pi,0.40940400673036015*pi) q[29];
U1q(3.277518916679441*pi,1.592486992396743*pi) q[30];
U1q(3.189213215274829*pi,0.8168436093913414*pi) q[31];
U1q(1.53361778274572*pi,1.061551650424423*pi) q[32];
U1q(3.288055551098763*pi,1.040954584008174*pi) q[33];
U1q(1.39482558449788*pi,1.275310572256061*pi) q[34];
U1q(1.82221821243046*pi,1.6054149604417627*pi) q[35];
U1q(1.32209163465196*pi,1.85678193261638*pi) q[36];
U1q(1.58026802863243*pi,0.888274172526268*pi) q[37];
U1q(0.236357913020531*pi,0.6310145018528899*pi) q[38];
U1q(3.843380067797691*pi,0.033100940413373525*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[2],q[13];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[6],q[21];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[14],q[9];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[12],q[22];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[32],q[17];
RZZ(0.5*pi) q[26],q[18];
RZZ(0.5*pi) q[28],q[20];
RZZ(0.5*pi) q[30],q[25];
RZZ(0.5*pi) q[27],q[34];
RZZ(0.5*pi) q[29],q[33];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[35],q[36];
U1q(1.8393403695228*pi,1.9319446424612168*pi) q[0];
U1q(1.64505752085989*pi,1.149644145787831*pi) q[1];
U1q(0.691303630057864*pi,0.749712257383313*pi) q[2];
U1q(1.37627639384002*pi,1.9385741054618157*pi) q[3];
U1q(1.8859164983053*pi,0.7703173243954133*pi) q[4];
U1q(1.30479668584512*pi,0.21342031128952818*pi) q[5];
U1q(1.59950671533256*pi,1.9863024805905218*pi) q[6];
U1q(3.362315452534769*pi,1.6021530084252413*pi) q[7];
U1q(3.21338934268116*pi,0.35365570226648557*pi) q[8];
U1q(1.72190875683943*pi,1.063795978809757*pi) q[9];
U1q(1.60885974936218*pi,0.3364388460859802*pi) q[10];
U1q(1.61701139110802*pi,1.3528867114040253*pi) q[11];
U1q(0.570967215139066*pi,0.6682605514199*pi) q[12];
U1q(3.562867817245325*pi,1.024189080219275*pi) q[13];
U1q(0.72486712172394*pi,0.42140000150780255*pi) q[14];
U1q(1.62539689095624*pi,1.60902570995168*pi) q[15];
U1q(3.678530227738262*pi,0.8414929266362878*pi) q[16];
U1q(0.67846389156235*pi,0.8904256078681974*pi) q[17];
U1q(1.16256744165816*pi,1.5158165617300345*pi) q[18];
U1q(1.89269653453426*pi,1.6543367542195204*pi) q[19];
U1q(3.587874316761516*pi,0.18491431211363318*pi) q[20];
U1q(1.69084917341512*pi,1.1050658882815698*pi) q[21];
U1q(0.727737707795885*pi,0.5982199502568801*pi) q[22];
U1q(1.27749546484411*pi,0.8673487527550403*pi) q[23];
U1q(1.20250380152745*pi,1.6135169565157899*pi) q[24];
U1q(0.723665678904415*pi,1.41438443202942*pi) q[25];
U1q(1.45945723447813*pi,1.56909567893603*pi) q[26];
U1q(0.696509977854147*pi,0.6835996092413701*pi) q[27];
U1q(3.224062158242833*pi,1.9667524676477814*pi) q[28];
U1q(3.624636750734207*pi,0.4945322259989302*pi) q[29];
U1q(3.507134003763694*pi,0.237311948206127*pi) q[30];
U1q(1.31978964213247*pi,0.9596092088216499*pi) q[31];
U1q(3.82682856158925*pi,0.5235884453430508*pi) q[32];
U1q(3.294900581446377*pi,1.3403016723845629*pi) q[33];
U1q(1.75309229601627*pi,1.7636699849016026*pi) q[34];
U1q(0.347025961785994*pi,0.5305868110001319*pi) q[35];
U1q(3.54722919623025*pi,0.965541559886983*pi) q[36];
U1q(0.491225778119674*pi,0.43399135158941204*pi) q[37];
U1q(1.16872514915377*pi,0.05231709047648003*pi) q[38];
U1q(3.246483897872848*pi,1.4698948989627034*pi) q[39];
RZZ(0.5*pi) q[0],q[24];
RZZ(0.5*pi) q[16],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[21];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[7],q[39];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[12],q[9];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[36],q[13];
RZZ(0.5*pi) q[31],q[14];
RZZ(0.5*pi) q[23],q[17];
RZZ(0.5*pi) q[27],q[19];
RZZ(0.5*pi) q[20],q[33];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[32],q[28];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[38],q[37];
U1q(3.255360276790193*pi,1.738093006860514*pi) q[0];
U1q(3.356976987931218*pi,0.47765301467933075*pi) q[1];
U1q(1.15266520456277*pi,0.507405237309773*pi) q[2];
U1q(0.302241658268885*pi,1.2046642083263608*pi) q[3];
U1q(3.543290316116014*pi,1.8441161200571212*pi) q[4];
U1q(1.20849189360178*pi,0.9795279992587262*pi) q[5];
U1q(1.41210352369922*pi,0.123710245516222*pi) q[6];
U1q(1.70763070923458*pi,1.8840237238247113*pi) q[7];
U1q(1.76531636662485*pi,1.2781974437079184*pi) q[8];
U1q(1.60137258872521*pi,0.47695366300131914*pi) q[9];
U1q(3.588152101591855*pi,0.8643255425483121*pi) q[10];
U1q(1.32671443257804*pi,1.3328646485485234*pi) q[11];
U1q(1.61166692635639*pi,1.8615543037151197*pi) q[12];
U1q(1.17282220721696*pi,1.0429164675060534*pi) q[13];
U1q(1.44797882562107*pi,1.333167172763563*pi) q[14];
U1q(1.54135939285309*pi,1.243460706464175*pi) q[15];
U1q(1.55306755643634*pi,1.7698844108148375*pi) q[16];
U1q(0.293183715594288*pi,0.34577711256636245*pi) q[17];
U1q(0.856584670723727*pi,1.0003353358925704*pi) q[18];
U1q(1.76944186616559*pi,0.07394026504023454*pi) q[19];
U1q(1.76522325475565*pi,1.6030046120611832*pi) q[20];
U1q(3.168777176986666*pi,0.7084250103789871*pi) q[21];
U1q(1.28695952714872*pi,0.6985919008459871*pi) q[22];
U1q(1.14305050157943*pi,1.5254100914776387*pi) q[23];
U1q(1.21568910344884*pi,1.0770348638374396*pi) q[24];
U1q(0.876326028192074*pi,1.00589847130047*pi) q[25];
U1q(1.53902812693247*pi,0.6807013572974152*pi) q[26];
U1q(3.88448422986685*pi,1.06083566796497*pi) q[27];
U1q(1.7913494335039*pi,1.123424092056188*pi) q[28];
U1q(1.90642184308048*pi,1.8315211617629306*pi) q[29];
U1q(1.82098972624752*pi,1.9161791528139034*pi) q[30];
U1q(1.74779905120078*pi,0.35548276190924*pi) q[31];
U1q(1.20335751203441*pi,1.3256077328137006*pi) q[32];
U1q(3.414278507483904*pi,1.573251547313463*pi) q[33];
U1q(0.533847643803775*pi,1.8677398444335367*pi) q[34];
U1q(1.82445879267217*pi,1.1213149138716498*pi) q[35];
U1q(1.328594092039*pi,0.3543737015127304*pi) q[36];
U1q(0.835234043521264*pi,1.523328725156372*pi) q[37];
U1q(1.51712056471307*pi,0.019043300116835304*pi) q[38];
U1q(1.06786120875044*pi,0.12564958561764916*pi) q[39];
RZZ(0.5*pi) q[0],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[31],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[16],q[11];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[38],q[14];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[20],q[25];
RZZ(0.5*pi) q[24],q[21];
RZZ(0.5*pi) q[27],q[22];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[28],q[36];
RZZ(0.5*pi) q[30],q[37];
RZZ(0.5*pi) q[39],q[32];
U1q(1.38878185785926*pi,0.376712730990465*pi) q[0];
U1q(1.8198315015968*pi,0.754013998010105*pi) q[1];
U1q(3.383697405403734*pi,1.669350161898962*pi) q[2];
U1q(0.803799036979463*pi,0.12705689693242084*pi) q[3];
U1q(3.836869592374336*pi,0.6448821107489201*pi) q[4];
U1q(3.399595863543094*pi,0.6478276851293896*pi) q[5];
U1q(1.53484413974085*pi,0.06740147907873606*pi) q[6];
U1q(3.199510200653075*pi,0.19601790071091818*pi) q[7];
U1q(3.0559071638511233*pi,1.1468718565218152*pi) q[8];
U1q(0.633727500499019*pi,1.6724215000981992*pi) q[9];
U1q(0.734092058430003*pi,0.45880116268511184*pi) q[10];
U1q(1.44230305628624*pi,0.46409544148399795*pi) q[11];
U1q(1.59123671288135*pi,0.29674665592254446*pi) q[12];
U1q(1.8840625052754*pi,0.40865438893191364*pi) q[13];
U1q(3.421380214019622*pi,1.3066111248372403*pi) q[14];
U1q(0.640387843921952*pi,0.30833686329599486*pi) q[15];
U1q(1.50527427353963*pi,1.2942935900833628*pi) q[16];
U1q(0.282085280080274*pi,1.4710862811234673*pi) q[17];
U1q(1.70102276081333*pi,0.20209444655043063*pi) q[18];
U1q(1.20612172112506*pi,1.4583411807333437*pi) q[19];
U1q(0.531653062334403*pi,0.6911605234784131*pi) q[20];
U1q(0.451197904780388*pi,1.8516340580207267*pi) q[21];
U1q(1.43879664400543*pi,1.4436697094537925*pi) q[22];
U1q(0.657976658093836*pi,1.6985708625576583*pi) q[23];
U1q(0.830528619964291*pi,0.08027433685707974*pi) q[24];
U1q(0.472941551538038*pi,1.45575988875524*pi) q[25];
U1q(0.778343509509143*pi,0.07205223218841539*pi) q[26];
U1q(1.54684669159577*pi,0.2562928053793274*pi) q[27];
U1q(3.4599468109998153*pi,1.9673980629104877*pi) q[28];
U1q(0.284363806712946*pi,1.2865508641131314*pi) q[29];
U1q(3.727420487609067*pi,0.11491834619452534*pi) q[30];
U1q(3.420599360288807*pi,0.0972223286072329*pi) q[31];
U1q(1.57348357643049*pi,1.663452429787311*pi) q[32];
U1q(3.028383726091588*pi,1.1397130151796127*pi) q[33];
U1q(1.24350114170571*pi,0.3249564209012519*pi) q[34];
U1q(1.90141982567879*pi,1.7911463741775222*pi) q[35];
U1q(0.820597456415735*pi,0.5440530256731604*pi) q[36];
U1q(1.12137766703766*pi,1.311400837193812*pi) q[37];
U1q(0.568149189557907*pi,0.5068294296469853*pi) q[38];
U1q(1.89246473578451*pi,1.020060984981365*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[1],q[17];
RZZ(0.5*pi) q[12],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[7],q[6];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[39],q[10];
RZZ(0.5*pi) q[35],q[11];
RZZ(0.5*pi) q[38],q[13];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[31],q[20];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[26],q[33];
RZZ(0.5*pi) q[27],q[37];
U1q(3.565584274931472*pi,1.5539512377720852*pi) q[0];
U1q(0.533611246473373*pi,0.3427320774716751*pi) q[1];
U1q(1.82309377452698*pi,0.38918129994487627*pi) q[2];
U1q(3.302583496012672*pi,0.950895142142131*pi) q[3];
U1q(3.28282782362775*pi,1.592580904098894*pi) q[4];
U1q(1.64557993593859*pi,1.6407434911381*pi) q[5];
U1q(1.81578563336093*pi,0.6594880686261857*pi) q[6];
U1q(3.3238872552737773*pi,0.2645378192905885*pi) q[7];
U1q(3.059216547057184*pi,0.0917708264613053*pi) q[8];
U1q(1.43771543282503*pi,0.05823621147078928*pi) q[9];
U1q(3.309756828409122*pi,0.504649286262822*pi) q[10];
U1q(3.290560887431452*pi,0.9168610000565574*pi) q[11];
U1q(0.595898350306838*pi,1.7711489337858248*pi) q[12];
U1q(3.672892850898536*pi,1.4044404114376605*pi) q[13];
U1q(3.546839469837058*pi,0.7258571185796008*pi) q[14];
U1q(3.220745913417157*pi,0.23859387952169442*pi) q[15];
U1q(0.1972847372104*pi,1.6521996467889126*pi) q[16];
U1q(1.56094549301791*pi,1.2568455430133678*pi) q[17];
U1q(3.884219453659948*pi,1.6556343239803484*pi) q[18];
U1q(3.838903991921998*pi,1.1851995970867382*pi) q[19];
U1q(0.799809319061094*pi,0.4193366684063431*pi) q[20];
U1q(1.56284103989177*pi,0.7944551983889667*pi) q[21];
U1q(1.62815823886124*pi,1.6212720019165223*pi) q[22];
U1q(0.125520738099871*pi,1.9691374894591878*pi) q[23];
U1q(1.76843639980003*pi,1.6566261829308102*pi) q[24];
U1q(1.68581849146081*pi,1.547951499993804*pi) q[25];
U1q(1.68072876159311*pi,1.9002844719905951*pi) q[26];
U1q(0.0767048656117201*pi,1.4792060826175275*pi) q[27];
U1q(1.4479863529468*pi,0.7273719295496663*pi) q[28];
U1q(1.50318120867368*pi,0.054899963402711194*pi) q[29];
U1q(1.09656407104518*pi,0.6237965286606943*pi) q[30];
U1q(3.735719479176231*pi,1.854520002952833*pi) q[31];
U1q(1.25387536955505*pi,0.31716848390585994*pi) q[32];
U1q(3.43180423401442*pi,1.902274685169293*pi) q[33];
U1q(1.61629314184491*pi,1.3081172243945647*pi) q[34];
U1q(0.498309037622699*pi,0.2251790158800342*pi) q[35];
U1q(1.29665620692194*pi,0.7991709699772604*pi) q[36];
U1q(1.62422525952732*pi,1.2101574765196847*pi) q[37];
U1q(0.0279513113714409*pi,1.7476833519727144*pi) q[38];
U1q(3.598457910894625*pi,0.4126038966355656*pi) q[39];
RZZ(0.5*pi) q[0],q[32];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[20],q[5];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[8],q[13];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[27],q[10];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[30],q[12];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[15],q[34];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[39],q[21];
RZZ(0.5*pi) q[38],q[25];
RZZ(0.5*pi) q[31],q[28];
RZZ(0.5*pi) q[36],q[37];
U1q(1.28476840443643*pi,1.3821003387349418*pi) q[0];
U1q(0.177612618811872*pi,1.4137464961169748*pi) q[1];
U1q(0.521761369246755*pi,1.437877593919536*pi) q[2];
U1q(1.32477060936149*pi,0.18630713411735877*pi) q[3];
U1q(3.396445174238741*pi,0.7956887159466407*pi) q[4];
U1q(0.76022306291728*pi,0.7943126191446819*pi) q[5];
U1q(1.7152325310773*pi,1.6904019594192534*pi) q[6];
U1q(1.51215961605759*pi,1.4073108289007585*pi) q[7];
U1q(1.7497852753213*pi,0.6550354516343413*pi) q[8];
U1q(1.73354701194431*pi,1.339744208317402*pi) q[9];
U1q(1.15705675370197*pi,1.6180233256943133*pi) q[10];
U1q(1.58569856112054*pi,0.30014868670474293*pi) q[11];
U1q(0.276751529625024*pi,1.112545635931765*pi) q[12];
U1q(1.38353013003044*pi,1.5359536481685456*pi) q[13];
U1q(1.55284706236463*pi,0.3910615147716401*pi) q[14];
U1q(3.9125895795915904*pi,0.6185495481768424*pi) q[15];
U1q(0.709439635201036*pi,1.5291080533029717*pi) q[16];
U1q(1.55500404607851*pi,0.22512368261937432*pi) q[17];
U1q(0.428670539148318*pi,0.9547485568966083*pi) q[18];
U1q(1.34018685057662*pi,1.0642951397304596*pi) q[19];
U1q(0.757882507629551*pi,1.9458356041850227*pi) q[20];
U1q(1.78245236458621*pi,0.6038041042280726*pi) q[21];
U1q(1.41291383984526*pi,1.8750746737960355*pi) q[22];
U1q(0.133365104753342*pi,0.319122702472999*pi) q[23];
U1q(1.56294963102449*pi,0.33137468471650955*pi) q[24];
U1q(1.67873095632485*pi,0.22666817752324042*pi) q[25];
U1q(1.559657612391*pi,0.9517312764243773*pi) q[26];
U1q(0.539729310596461*pi,0.6807196936838875*pi) q[27];
U1q(0.588193532354429*pi,1.0125178403647865*pi) q[28];
U1q(1.80935401981985*pi,0.1848541076804473*pi) q[29];
U1q(0.27675961794917*pi,0.5661265284897947*pi) q[30];
U1q(1.78703550405397*pi,1.6012338225224982*pi) q[31];
U1q(0.589694067604941*pi,0.5052821681418704*pi) q[32];
U1q(1.59766124961709*pi,0.8911186317308761*pi) q[33];
U1q(0.276695975819652*pi,0.824631087536984*pi) q[34];
U1q(0.565465802329635*pi,1.4236568117622141*pi) q[35];
U1q(1.53602901629055*pi,1.3252384430912958*pi) q[36];
U1q(0.791080946318617*pi,1.4645226349058849*pi) q[37];
U1q(0.470631269293164*pi,1.505530213208754*pi) q[38];
U1q(1.6592556120199*pi,0.5003237330821282*pi) q[39];
rz(2.6178996612650582*pi) q[0];
rz(0.5862535038830252*pi) q[1];
rz(2.562122406080464*pi) q[2];
rz(3.8136928658826412*pi) q[3];
rz(3.204311284053359*pi) q[4];
rz(3.205687380855318*pi) q[5];
rz(0.30959804058074647*pi) q[6];
rz(0.5926891710992415*pi) q[7];
rz(3.3449645483656587*pi) q[8];
rz(0.660255791682598*pi) q[9];
rz(2.3819766743056867*pi) q[10];
rz(1.699851313295257*pi) q[11];
rz(0.8874543640682351*pi) q[12];
rz(2.4640463518314544*pi) q[13];
rz(3.60893848522836*pi) q[14];
rz(1.3814504518231576*pi) q[15];
rz(0.4708919466970283*pi) q[16];
rz(3.7748763173806257*pi) q[17];
rz(3.0452514431033917*pi) q[18];
rz(0.9357048602695404*pi) q[19];
rz(2.0541643958149773*pi) q[20];
rz(1.3961958957719274*pi) q[21];
rz(2.1249253262039645*pi) q[22];
rz(1.680877297527001*pi) q[23];
rz(3.6686253152834905*pi) q[24];
rz(3.7733318224767594*pi) q[25];
rz(1.0482687235756227*pi) q[26];
rz(3.3192803063161125*pi) q[27];
rz(0.9874821596352135*pi) q[28];
rz(1.8151458923195527*pi) q[29];
rz(3.4338734715102053*pi) q[30];
rz(2.3987661774775018*pi) q[31];
rz(1.4947178318581296*pi) q[32];
rz(1.1088813682691239*pi) q[33];
rz(3.175368912463016*pi) q[34];
rz(0.5763431882377859*pi) q[35];
rz(2.674761556908704*pi) q[36];
rz(0.5354773650941151*pi) q[37];
rz(2.494469786791246*pi) q[38];
rz(3.499676266917872*pi) q[39];
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
