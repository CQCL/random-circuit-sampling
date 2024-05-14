OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.834176059904628*pi,1.594572450799951*pi) q[0];
U1q(1.17087154408203*pi,0.11636890602370957*pi) q[1];
U1q(1.5101069855812*pi,1.7452738573587079*pi) q[2];
U1q(3.266460557250395*pi,1.1597757602931689*pi) q[3];
U1q(3.432008256596483*pi,0.8930061021139367*pi) q[4];
U1q(1.58848930283292*pi,0.6122010359539755*pi) q[5];
U1q(0.41721672556941*pi,1.226244973400602*pi) q[6];
U1q(1.46967145677415*pi,0.49339367706749276*pi) q[7];
U1q(0.259774843032516*pi,0.332656889503031*pi) q[8];
U1q(0.280644361147323*pi,0.262709176691264*pi) q[9];
U1q(1.57566514559205*pi,0.3788983601891618*pi) q[10];
U1q(0.449267732419716*pi,0.9090460838835599*pi) q[11];
U1q(0.620648256542887*pi,0.629650158511188*pi) q[12];
U1q(0.554901280062903*pi,0.130520666617399*pi) q[13];
U1q(1.60202570670049*pi,0.9221065635449454*pi) q[14];
U1q(1.47735385403098*pi,1.0850795071126726*pi) q[15];
U1q(0.388710924050774*pi,0.190477724533418*pi) q[16];
U1q(3.618070392989472*pi,0.5208414041426718*pi) q[17];
U1q(1.63200620055618*pi,0.17996430688868062*pi) q[18];
U1q(1.21112718417097*pi,0.9442589863916907*pi) q[19];
U1q(0.221643032864053*pi,1.4570305285198288*pi) q[20];
U1q(3.422228139330154*pi,0.9212741883333463*pi) q[21];
U1q(3.317350348329762*pi,1.3012164186382815*pi) q[22];
U1q(1.77851006227042*pi,1.4631134364065157*pi) q[23];
U1q(1.24927452155187*pi,0.3850302015682464*pi) q[24];
U1q(1.79194122495056*pi,0.25493097549962235*pi) q[25];
U1q(1.5672992390569*pi,1.82393546323644*pi) q[26];
U1q(0.53619643664084*pi,1.666528357652353*pi) q[27];
U1q(0.264446222403785*pi,0.245185352736528*pi) q[28];
U1q(0.709059922261463*pi,1.9068195900985663*pi) q[29];
U1q(0.665650641437571*pi,0.588381693250391*pi) q[30];
U1q(0.76068376713844*pi,0.339366329016573*pi) q[31];
U1q(0.167200125929291*pi,0.218924333906819*pi) q[32];
U1q(0.555325407733952*pi,0.248632082884939*pi) q[33];
U1q(1.76117466940366*pi,1.5597113906271505*pi) q[34];
U1q(1.2261364332813*pi,0.6770224590866163*pi) q[35];
U1q(1.36384469698949*pi,0.29110357204183696*pi) q[36];
U1q(1.61808174302692*pi,1.4154467007959561*pi) q[37];
U1q(3.806809129744473*pi,1.1508130863211357*pi) q[38];
U1q(1.47101068480051*pi,1.2828236263030042*pi) q[39];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[1],q[19];
RZZ(0.5*pi) q[2],q[17];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[28],q[4];
RZZ(0.5*pi) q[5],q[34];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[39],q[7];
RZZ(0.5*pi) q[9],q[29];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[15],q[13];
RZZ(0.5*pi) q[37],q[14];
RZZ(0.5*pi) q[36],q[18];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[21],q[25];
RZZ(0.5*pi) q[24],q[32];
RZZ(0.5*pi) q[31],q[26];
RZZ(0.5*pi) q[38],q[30];
RZZ(0.5*pi) q[33],q[35];
U1q(0.870976949829118*pi,0.47916181184286*pi) q[0];
U1q(0.671830596582772*pi,1.0032055084411393*pi) q[1];
U1q(0.415463600414051*pi,1.1097585600170978*pi) q[2];
U1q(0.529266609579656*pi,0.7192957441241292*pi) q[3];
U1q(0.133376495556242*pi,1.1283895153216466*pi) q[4];
U1q(0.465740717005879*pi,1.6140755730192757*pi) q[5];
U1q(0.601619576103573*pi,1.3413532796887901*pi) q[6];
U1q(0.661433644395826*pi,1.7516936073108598*pi) q[7];
U1q(0.572984456005634*pi,1.80253139076029*pi) q[8];
U1q(0.529370317616229*pi,1.86164948931612*pi) q[9];
U1q(0.666482998754483*pi,1.3746898885627816*pi) q[10];
U1q(0.508831928198279*pi,0.18217137698025*pi) q[11];
U1q(0.391882721007875*pi,1.1413295591208201*pi) q[12];
U1q(0.58548948615713*pi,1.3230369305684402*pi) q[13];
U1q(0.589970772184956*pi,1.945879385212915*pi) q[14];
U1q(0.336050834448551*pi,0.2976649599597425*pi) q[15];
U1q(0.569778683860882*pi,0.027614284230049968*pi) q[16];
U1q(0.805318872518027*pi,1.3475304909094317*pi) q[17];
U1q(0.200822093692791*pi,1.6234541742763406*pi) q[18];
U1q(0.65409931122031*pi,0.33010240911654076*pi) q[19];
U1q(0.139160493694862*pi,1.66329483548967*pi) q[20];
U1q(0.765058084486903*pi,1.4365611469169464*pi) q[21];
U1q(0.593154004024094*pi,0.46864931673333365*pi) q[22];
U1q(0.874944097910228*pi,0.6814079914964357*pi) q[23];
U1q(0.769106526129363*pi,1.4117765270524014*pi) q[24];
U1q(0.610188548594279*pi,0.04542259536245208*pi) q[25];
U1q(0.553460915587066*pi,1.62068610601023*pi) q[26];
U1q(0.362591543561355*pi,1.5188502049437496*pi) q[27];
U1q(0.545233091026874*pi,0.39527838714283003*pi) q[28];
U1q(0.685452144291428*pi,0.7520784601269801*pi) q[29];
U1q(0.494343204020847*pi,1.5798579148802898*pi) q[30];
U1q(0.403566070361848*pi,0.4726480405556901*pi) q[31];
U1q(0.517216511617564*pi,1.8685508597427298*pi) q[32];
U1q(0.506631914051728*pi,0.22102993619700007*pi) q[33];
U1q(0.431565205833987*pi,1.5797087458198305*pi) q[34];
U1q(0.814012524046174*pi,0.7997689094423164*pi) q[35];
U1q(0.282394917343339*pi,0.35794972602069697*pi) q[36];
U1q(0.578950301219937*pi,0.8546906331645263*pi) q[37];
U1q(0.215842768425269*pi,1.3866083057235556*pi) q[38];
U1q(0.978832892726277*pi,1.888046882121234*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[20],q[1];
RZZ(0.5*pi) q[2],q[29];
RZZ(0.5*pi) q[3],q[16];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[5],q[17];
RZZ(0.5*pi) q[28],q[6];
RZZ(0.5*pi) q[37],q[8];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[19],q[27];
RZZ(0.5*pi) q[21],q[26];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[30],q[31];
RZZ(0.5*pi) q[38],q[34];
U1q(0.277393336321557*pi,0.8705486009590597*pi) q[0];
U1q(0.658254312199599*pi,0.18244035956947968*pi) q[1];
U1q(0.300993204398591*pi,0.6087323599567274*pi) q[2];
U1q(0.563458338018215*pi,1.5679817761872092*pi) q[3];
U1q(0.591821599423527*pi,1.1354014186644967*pi) q[4];
U1q(0.315254377230582*pi,0.39813973799793523*pi) q[5];
U1q(0.524131917529042*pi,0.6638642097642302*pi) q[6];
U1q(0.824693229833522*pi,1.800706994899603*pi) q[7];
U1q(0.702428356760768*pi,0.1485956247588498*pi) q[8];
U1q(0.158420606129063*pi,0.6231374275097998*pi) q[9];
U1q(0.310449608386483*pi,1.7038767386060911*pi) q[10];
U1q(0.466592645762156*pi,0.86614039652956*pi) q[11];
U1q(0.360950984309206*pi,0.4233107990868499*pi) q[12];
U1q(0.202671530980061*pi,1.1882642212222896*pi) q[13];
U1q(0.372181308998477*pi,1.7634278672584855*pi) q[14];
U1q(0.122541631309191*pi,1.538482354999653*pi) q[15];
U1q(0.551967495953633*pi,1.4285636252011402*pi) q[16];
U1q(0.592475851304393*pi,1.0777322991025016*pi) q[17];
U1q(0.679171289356186*pi,0.3585751862795705*pi) q[18];
U1q(0.286674246537244*pi,1.1687537104718606*pi) q[19];
U1q(0.389497345970936*pi,1.2977670530156598*pi) q[20];
U1q(0.336957809745171*pi,0.7320219013602163*pi) q[21];
U1q(0.658521054241594*pi,1.0745720910840517*pi) q[22];
U1q(0.492609876894042*pi,0.21450007259927517*pi) q[23];
U1q(0.360122977076114*pi,1.3275374180942667*pi) q[24];
U1q(0.52621822192973*pi,0.7319822776455225*pi) q[25];
U1q(0.613048037812987*pi,0.4700139637357399*pi) q[26];
U1q(0.60118365660988*pi,0.17544723198532974*pi) q[27];
U1q(0.515866547075777*pi,1.4536568284654998*pi) q[28];
U1q(0.985183697864952*pi,0.1469809612973898*pi) q[29];
U1q(0.951634419775222*pi,0.6030161467964499*pi) q[30];
U1q(0.595838479978491*pi,0.1049972632765801*pi) q[31];
U1q(0.359310859277499*pi,0.15046811398291027*pi) q[32];
U1q(0.470345992628486*pi,0.8761989917698303*pi) q[33];
U1q(0.597894512189767*pi,0.4565699192191408*pi) q[34];
U1q(0.298779450269963*pi,0.45810221725413625*pi) q[35];
U1q(0.658712718188624*pi,0.7250996810932371*pi) q[36];
U1q(0.325340838559436*pi,1.2290107418573868*pi) q[37];
U1q(0.498506390670721*pi,0.5505215607998055*pi) q[38];
U1q(0.455194168286446*pi,1.3962337362583739*pi) q[39];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[39],q[1];
RZZ(0.5*pi) q[2],q[26];
RZZ(0.5*pi) q[3],q[28];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[31],q[7];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[11],q[23];
RZZ(0.5*pi) q[12],q[30];
RZZ(0.5*pi) q[33],q[13];
RZZ(0.5*pi) q[14],q[29];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[36],q[19];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[24],q[27];
RZZ(0.5*pi) q[25],q[32];
U1q(0.676081099228754*pi,0.7954773151456394*pi) q[0];
U1q(0.675605041197677*pi,1.5970846598137491*pi) q[1];
U1q(0.278207100097979*pi,0.8416003017099873*pi) q[2];
U1q(0.43336518513646*pi,1.6827552352164492*pi) q[3];
U1q(0.489307684333125*pi,1.5132160065209463*pi) q[4];
U1q(0.286631602552331*pi,1.4409751227791556*pi) q[5];
U1q(0.444539557241128*pi,1.3139004818629605*pi) q[6];
U1q(0.417827459306031*pi,1.6324642349037628*pi) q[7];
U1q(0.294506695550479*pi,1.84832523668128*pi) q[8];
U1q(0.562631391403411*pi,0.3638676231840101*pi) q[9];
U1q(0.175165868360638*pi,0.5038273464326419*pi) q[10];
U1q(0.465830041062983*pi,0.13821450962035975*pi) q[11];
U1q(0.021956138092236*pi,1.91716845334898*pi) q[12];
U1q(0.576985582683792*pi,0.8125289787563501*pi) q[13];
U1q(0.321405954709426*pi,0.2643585826796446*pi) q[14];
U1q(0.454308098677121*pi,1.1531211455321628*pi) q[15];
U1q(0.425472349357211*pi,0.38768165220179984*pi) q[16];
U1q(0.518058253209304*pi,0.43307947122556145*pi) q[17];
U1q(0.579193015120138*pi,1.762935431437601*pi) q[18];
U1q(0.20527389108793*pi,1.252634775449991*pi) q[19];
U1q(0.44859874801418*pi,0.9750847212232099*pi) q[20];
U1q(0.844852891022222*pi,1.965798722033397*pi) q[21];
U1q(0.407625071896896*pi,0.6571863234278013*pi) q[22];
U1q(0.541476342983989*pi,0.023146493133515023*pi) q[23];
U1q(0.651221021628685*pi,1.1301308691284557*pi) q[24];
U1q(0.495514715467137*pi,0.6921769723988023*pi) q[25];
U1q(0.203529523067833*pi,1.0877861193355294*pi) q[26];
U1q(0.76072797731513*pi,0.8439733197655599*pi) q[27];
U1q(0.166013715909436*pi,0.6886381215355497*pi) q[28];
U1q(0.647649205755293*pi,0.4036872127980704*pi) q[29];
U1q(0.546552020113083*pi,1.8681821281645004*pi) q[30];
U1q(0.443452528138513*pi,0.39767524069891014*pi) q[31];
U1q(0.810269396361268*pi,1.1921666304568204*pi) q[32];
U1q(0.252698810858139*pi,0.15143936516578993*pi) q[33];
U1q(0.446523224626165*pi,0.6825640953914505*pi) q[34];
U1q(0.69337362551196*pi,0.2327100583662265*pi) q[35];
U1q(0.820298886408463*pi,0.9514287947219264*pi) q[36];
U1q(0.525945812909679*pi,0.9933760883861549*pi) q[37];
U1q(0.23582378500418*pi,1.7049061978166655*pi) q[38];
U1q(0.425835724838268*pi,0.6725666499288643*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[1],q[28];
RZZ(0.5*pi) q[2],q[31];
RZZ(0.5*pi) q[3],q[10];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[29],q[7];
RZZ(0.5*pi) q[8],q[17];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[11],q[26];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[37];
RZZ(0.5*pi) q[15],q[38];
RZZ(0.5*pi) q[27],q[18];
RZZ(0.5*pi) q[25],q[20];
RZZ(0.5*pi) q[21],q[32];
RZZ(0.5*pi) q[23],q[30];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[33],q[34];
U1q(0.564977094492132*pi,0.3001269521493999*pi) q[0];
U1q(0.517267621685187*pi,1.8704194943596093*pi) q[1];
U1q(0.015734293297636*pi,1.1989922161874276*pi) q[2];
U1q(0.142612532992165*pi,1.1162974485875683*pi) q[3];
U1q(0.336676123896364*pi,1.739538543431987*pi) q[4];
U1q(0.284667302621947*pi,0.9033615325483755*pi) q[5];
U1q(0.770502519198927*pi,0.13103607956360008*pi) q[6];
U1q(0.869166092092779*pi,1.6392607072975727*pi) q[7];
U1q(0.607545343961726*pi,0.8672230273616801*pi) q[8];
U1q(0.178857730040388*pi,0.61321159024331*pi) q[9];
U1q(0.459341870815966*pi,1.787439141021462*pi) q[10];
U1q(0.704274342740458*pi,1.5457966066491*pi) q[11];
U1q(0.680207810371726*pi,1.1087557998479003*pi) q[12];
U1q(0.61265417027207*pi,1.9599256821250002*pi) q[13];
U1q(0.517235628766242*pi,1.2363667901371453*pi) q[14];
U1q(0.584782363544713*pi,1.0046714648752726*pi) q[15];
U1q(0.898613557113997*pi,1.7992804336013002*pi) q[16];
U1q(0.887534690016034*pi,1.3635093700468017*pi) q[17];
U1q(0.798670908379317*pi,0.3433334992327506*pi) q[18];
U1q(0.753643893382828*pi,0.446266235485111*pi) q[19];
U1q(0.568751471819469*pi,1.2330004463744597*pi) q[20];
U1q(0.113041476133712*pi,1.256247020320286*pi) q[21];
U1q(0.348076591942951*pi,1.9052200398699615*pi) q[22];
U1q(0.388733295227898*pi,1.7153754475061955*pi) q[23];
U1q(0.644343290296223*pi,0.7185939721587662*pi) q[24];
U1q(0.542889569394946*pi,0.9511283791988223*pi) q[25];
U1q(0.388488845028036*pi,0.5120989316968405*pi) q[26];
U1q(0.154537837417552*pi,0.6507156637511997*pi) q[27];
U1q(0.536586868197893*pi,1.2877116862160003*pi) q[28];
U1q(0.61968461333707*pi,0.7040467899494001*pi) q[29];
U1q(0.603114221068498*pi,0.5326703536039501*pi) q[30];
U1q(0.647228652480262*pi,1.6609658609973899*pi) q[31];
U1q(0.80697527863881*pi,0.8082434624326602*pi) q[32];
U1q(0.357243854913532*pi,0.5118649572644305*pi) q[33];
U1q(0.297813334940305*pi,0.30182157541774046*pi) q[34];
U1q(0.899607037653937*pi,1.8113241316346063*pi) q[35];
U1q(0.255989206570616*pi,1.2392850976367669*pi) q[36];
U1q(0.572889952838379*pi,0.7434845680532565*pi) q[37];
U1q(0.537504523806193*pi,1.7213919054396172*pi) q[38];
U1q(0.543247416197119*pi,1.8245387279755843*pi) q[39];
rz(0.6199637114672996*pi) q[0];
rz(1.1760518519343819*pi) q[1];
rz(1.6239461783412619*pi) q[2];
rz(1.7050634929745705*pi) q[3];
rz(0.42559122524593285*pi) q[4];
rz(3.332629654762024*pi) q[5];
rz(0.3978069611613009*pi) q[6];
rz(2.5530773551044774*pi) q[7];
rz(2.5869191178124*pi) q[8];
rz(1.7140704715141108*pi) q[9];
rz(3.7211704207449383*pi) q[10];
rz(2.8907548279984*pi) q[11];
rz(3.1854352076785997*pi) q[12];
rz(2.8428066585896996*pi) q[13];
rz(2.9747983376363543*pi) q[14];
rz(2.0040899205355274*pi) q[15];
rz(3.6232015229913*pi) q[16];
rz(1.6119429685527082*pi) q[17];
rz(2.7930545721747793*pi) q[18];
rz(3.9447502806766286*pi) q[19];
rz(1.6753246152187806*pi) q[20];
rz(3.705735375447513*pi) q[21];
rz(0.37129685114220834*pi) q[22];
rz(2.4548934851290447*pi) q[23];
rz(2.0547707271355042*pi) q[24];
rz(1.928312260326777*pi) q[25];
rz(3.2423736781922603*pi) q[26];
rz(3.8523394936923996*pi) q[27];
rz(2.764287432532*pi) q[28];
rz(3.0499945896172003*pi) q[29];
rz(2.04262731324497*pi) q[30];
rz(1.0219280452892008*pi) q[31];
rz(2.6787713683477596*pi) q[32];
rz(1.1464051858718207*pi) q[33];
rz(1.3013068888517694*pi) q[34];
rz(0.43623404001914245*pi) q[35];
rz(3.3876481834642433*pi) q[36];
rz(0.4740671254974451*pi) q[37];
rz(2.7694704952988634*pi) q[38];
rz(1.8820363398255253*pi) q[39];
U1q(1.56497709449213*pi,1.9200906636166435*pi) q[0];
U1q(0.517267621685187*pi,0.0464713462939499*pi) q[1];
U1q(0.015734293297636*pi,1.822938394528686*pi) q[2];
U1q(0.142612532992165*pi,1.821360941562139*pi) q[3];
U1q(0.336676123896364*pi,1.165129768677918*pi) q[4];
U1q(0.284667302621947*pi,1.235991187310409*pi) q[5];
U1q(0.770502519198927*pi,1.52884304072489*pi) q[6];
U1q(0.869166092092779*pi,1.19233806240205*pi) q[7];
U1q(0.607545343961726*pi,0.454142145174077*pi) q[8];
U1q(0.178857730040388*pi,1.327282061757419*pi) q[9];
U1q(0.459341870815966*pi,0.5086095617664601*pi) q[10];
U1q(0.704274342740458*pi,1.436551434647488*pi) q[11];
U1q(3.680207810371726*pi,1.294191007526512*pi) q[12];
U1q(1.61265417027207*pi,1.802732340714666*pi) q[13];
U1q(1.51723562876624*pi,1.211165127773544*pi) q[14];
U1q(0.584782363544713*pi,0.00876138541075289*pi) q[15];
U1q(0.898613557113997*pi,0.422481956592602*pi) q[16];
U1q(3.887534690016034*pi,1.9754523385995169*pi) q[17];
U1q(0.798670908379317*pi,0.136388071407527*pi) q[18];
U1q(0.753643893382828*pi,1.39101651616174*pi) q[19];
U1q(0.568751471819469*pi,1.9083250615932352*pi) q[20];
U1q(1.11304147613371*pi,1.961982395767798*pi) q[21];
U1q(3.348076591942951*pi,1.276516891012169*pi) q[22];
U1q(0.388733295227898*pi,1.17026893263524*pi) q[23];
U1q(0.644343290296223*pi,1.773364699294271*pi) q[24];
U1q(0.542889569394946*pi,1.879440639525588*pi) q[25];
U1q(0.388488845028036*pi,0.754472609889081*pi) q[26];
U1q(0.154537837417552*pi,1.503055157443607*pi) q[27];
U1q(0.536586868197893*pi,1.051999118747994*pi) q[28];
U1q(0.61968461333707*pi,0.754041379566604*pi) q[29];
U1q(0.603114221068498*pi,1.575297666848915*pi) q[30];
U1q(1.64722865248026*pi,1.682893906286588*pi) q[31];
U1q(1.80697527863881*pi,0.487014830780418*pi) q[32];
U1q(0.357243854913532*pi,0.658270143136254*pi) q[33];
U1q(3.297813334940305*pi,0.603128464269507*pi) q[34];
U1q(0.899607037653937*pi,1.247558171653746*pi) q[35];
U1q(0.255989206570616*pi,1.626933281101011*pi) q[36];
U1q(0.572889952838379*pi,0.217551693550761*pi) q[37];
U1q(0.537504523806193*pi,1.490862400738477*pi) q[38];
U1q(0.543247416197119*pi,0.706575067801114*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[1],q[28];
RZZ(0.5*pi) q[2],q[31];
RZZ(0.5*pi) q[3],q[10];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[29],q[7];
RZZ(0.5*pi) q[8],q[17];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[11],q[26];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[37];
RZZ(0.5*pi) q[15],q[38];
RZZ(0.5*pi) q[27],q[18];
RZZ(0.5*pi) q[25],q[20];
RZZ(0.5*pi) q[21],q[32];
RZZ(0.5*pi) q[23],q[30];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[33],q[34];
U1q(3.323918900771246*pi,0.42474030062038615*pi) q[0];
U1q(1.67560504119768*pi,0.7731365117481301*pi) q[1];
U1q(3.278207100097979*pi,1.4655464800512399*pi) q[2];
U1q(0.43336518513646*pi,1.38781872819103*pi) q[3];
U1q(1.48930768433313*pi,1.93880723176688*pi) q[4];
U1q(1.28663160255233*pi,0.7736047775411901*pi) q[5];
U1q(0.444539557241128*pi,1.7117074430242498*pi) q[6];
U1q(3.417827459306031*pi,0.185541590008239*pi) q[7];
U1q(1.29450669555048*pi,1.435244354493681*pi) q[8];
U1q(0.562631391403411*pi,1.07793809469812*pi) q[9];
U1q(0.175165868360638*pi,0.22499776717761*pi) q[10];
U1q(1.46583004106298*pi,1.02896933761878*pi) q[11];
U1q(1.02195613809224*pi,1.48577835402545*pi) q[12];
U1q(3.423014417316208*pi,1.950129044083325*pi) q[13];
U1q(3.678594045290573*pi,1.183173335231064*pi) q[14];
U1q(1.45430809867712*pi,1.157211066067667*pi) q[15];
U1q(0.425472349357211*pi,0.010883175193139971*pi) q[16];
U1q(1.5180582532093*pi,0.9058822374207613*pi) q[17];
U1q(0.579193015120138*pi,0.5559900036123899*pi) q[18];
U1q(0.20527389108793*pi,0.19738505612661017*pi) q[19];
U1q(1.44859874801418*pi,0.6504093364419798*pi) q[20];
U1q(3.155147108977777*pi,1.25243069405469*pi) q[21];
U1q(3.592374928103104*pi,0.5245506074543343*pi) q[22];
U1q(0.541476342983989*pi,0.478039978262555*pi) q[23];
U1q(1.65122102162869*pi,1.18490159626396*pi) q[24];
U1q(1.49551471546714*pi,0.62048923272557*pi) q[25];
U1q(0.203529523067833*pi,0.33015979752774993*pi) q[26];
U1q(0.76072797731513*pi,0.69631281345793*pi) q[27];
U1q(0.166013715909436*pi,1.45292555406755*pi) q[28];
U1q(1.64764920575529*pi,1.4536818024152751*pi) q[29];
U1q(0.546552020113083*pi,1.9108094414094698*pi) q[30];
U1q(3.556547471861487*pi,1.946184526585057*pi) q[31];
U1q(1.81026939636127*pi,1.1030916627562517*pi) q[32];
U1q(1.25269881085814*pi,0.29784455103762*pi) q[33];
U1q(1.44652322462616*pi,0.22238594429579772*pi) q[34];
U1q(1.69337362551196*pi,0.66894409838536*pi) q[35];
U1q(1.82029888640846*pi,0.3390769781861702*pi) q[36];
U1q(1.52594581290968*pi,0.46744321388365995*pi) q[37];
U1q(1.23582378500418*pi,0.47437669311552*pi) q[38];
U1q(0.425835724838268*pi,0.55460298975439*pi) q[39];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[39],q[1];
RZZ(0.5*pi) q[2],q[26];
RZZ(0.5*pi) q[3],q[28];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[31],q[7];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[11],q[23];
RZZ(0.5*pi) q[12],q[30];
RZZ(0.5*pi) q[33],q[13];
RZZ(0.5*pi) q[14],q[29];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[36],q[19];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[24],q[27];
RZZ(0.5*pi) q[25],q[32];
U1q(3.722606663678443*pi,1.3496690148069646*pi) q[0];
U1q(3.341745687800401*pi,1.1877808119924071*pi) q[1];
U1q(3.699006795601409*pi,0.6984144218045039*pi) q[2];
U1q(0.563458338018215*pi,0.27304526916178995*pi) q[3];
U1q(3.408178400576473*pi,1.316621819623327*pi) q[4];
U1q(3.684745622769418*pi,0.8164401623223985*pi) q[5];
U1q(0.524131917529042*pi,1.06167117092552*pi) q[6];
U1q(1.82469322983352*pi,1.0172988300123933*pi) q[7];
U1q(3.2975716432392312*pi,1.1349739664161076*pi) q[8];
U1q(1.15842060612906*pi,1.3372078990239098*pi) q[9];
U1q(0.310449608386483*pi,1.42504715935106*pi) q[10];
U1q(3.533407354237844*pi,1.3010434507095803*pi) q[11];
U1q(1.36095098430921*pi,0.9919206997633196*pi) q[12];
U1q(1.20267153098006*pi,1.5743938016173853*pi) q[13];
U1q(1.37218130899848*pi,1.6841040506522291*pi) q[14];
U1q(3.122541631309191*pi,1.7718498566001815*pi) q[15];
U1q(1.55196749595363*pi,0.051765148192479904*pi) q[16];
U1q(0.592475851304393*pi,0.5505350652977041*pi) q[17];
U1q(0.679171289356186*pi,0.15162975845434978*pi) q[18];
U1q(0.286674246537244*pi,0.113503991148487*pi) q[19];
U1q(3.3894973459709368*pi,1.3277270046495318*pi) q[20];
U1q(1.33695780974517*pi,0.4862075147278615*pi) q[21];
U1q(1.65852105424159*pi,0.107164839798077*pi) q[22];
U1q(1.49260987689404*pi,0.669393557728314*pi) q[23];
U1q(1.36012297707611*pi,0.987495047298157*pi) q[24];
U1q(1.52621822192973*pi,0.5806839274788471*pi) q[25];
U1q(0.613048037812987*pi,1.7123876419279598*pi) q[26];
U1q(1.60118365660988*pi,0.0277867256776917*pi) q[27];
U1q(1.51586654707578*pi,1.2179442609975002*pi) q[28];
U1q(1.98518369786495*pi,0.7103880539159637*pi) q[29];
U1q(0.951634419775222*pi,1.6456434600414198*pi) q[30];
U1q(3.4041615200215087*pi,1.238862504007387*pi) q[31];
U1q(1.3593108592775*pi,0.061393146282341826*pi) q[32];
U1q(3.470345992628486*pi,0.5730849244335843*pi) q[33];
U1q(3.597894512189767*pi,0.9963917681234907*pi) q[34];
U1q(1.29877945026996*pi,1.4435519394974448*pi) q[35];
U1q(3.341287281811376*pi,1.5654060918148582*pi) q[36];
U1q(1.32534083855944*pi,0.23180856041245113*pi) q[37];
U1q(3.501493609329279*pi,1.62876133013238*pi) q[38];
U1q(3.4551941682864458*pi,1.2782700760839*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[20],q[1];
RZZ(0.5*pi) q[2],q[29];
RZZ(0.5*pi) q[3],q[16];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[5],q[17];
RZZ(0.5*pi) q[28],q[6];
RZZ(0.5*pi) q[37],q[8];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[19],q[27];
RZZ(0.5*pi) q[21],q[26];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[30],q[31];
RZZ(0.5*pi) q[38],q[34];
U1q(1.87097694982912*pi,1.7410558039231676*pi) q[0];
U1q(3.328169403417228*pi,0.3670156631207466*pi) q[1];
U1q(3.415463600414051*pi,1.197388221744132*pi) q[2];
U1q(1.52926660957966*pi,1.4243592370987104*pi) q[3];
U1q(1.13337649555624*pi,0.3236337229661783*pi) q[4];
U1q(1.46574071700588*pi,0.6005043273010715*pi) q[5];
U1q(1.60161957610357*pi,1.7391602408500804*pi) q[6];
U1q(1.66143364439583*pi,0.9682854424236471*pi) q[7];
U1q(3.427015543994366*pi,1.4810382004146776*pi) q[8];
U1q(3.470629682383771*pi,0.09869583721758701*pi) q[9];
U1q(1.66648299875448*pi,0.09586030930773992*pi) q[10];
U1q(3.491168071801721*pi,1.9850124702588872*pi) q[11];
U1q(3.608117278992124*pi,0.27390193972934984*pi) q[12];
U1q(0.58548948615713*pi,0.7091665109635272*pi) q[13];
U1q(1.58997077218496*pi,0.8665555686066599*pi) q[14];
U1q(3.336050834448551*pi,0.5310324615602715*pi) q[15];
U1q(3.569778683860882*pi,1.4527144891635735*pi) q[16];
U1q(0.805318872518027*pi,0.8203332571046342*pi) q[17];
U1q(0.200822093692791*pi,1.41650874645112*pi) q[18];
U1q(0.65409931122031*pi,1.27485268979316*pi) q[19];
U1q(0.139160493694862*pi,1.693254787123542*pi) q[20];
U1q(1.7650580844869*pi,0.19074676028458137*pi) q[21];
U1q(1.59315400402409*pi,0.501242065447359*pi) q[22];
U1q(3.125055902089772*pi,0.20248563883115336*pi) q[23];
U1q(0.769106526129363*pi,0.07173415625629698*pi) q[24];
U1q(0.610188548594279*pi,0.8941242451957669*pi) q[25];
U1q(1.55346091558707*pi,1.8630597842024397*pi) q[26];
U1q(1.36259154356136*pi,0.6843837527192697*pi) q[27];
U1q(1.54523309102687*pi,0.2763227023201704*pi) q[28];
U1q(1.68545214429143*pi,1.3154855527455438*pi) q[29];
U1q(0.494343204020847*pi,0.6224852281252602*pi) q[30];
U1q(1.40356607036185*pi,1.8712117267282764*pi) q[31];
U1q(1.51721651161756*pi,1.3433104005225205*pi) q[32];
U1q(1.50663191405173*pi,0.9179158688607547*pi) q[33];
U1q(3.568434794166013*pi,1.8732529415228083*pi) q[34];
U1q(0.814012524046174*pi,0.7852186316856147*pi) q[35];
U1q(1.28239491734334*pi,1.9325560468873997*pi) q[36];
U1q(1.57895030121994*pi,1.8574884517195915*pi) q[37];
U1q(1.21584276842527*pi,0.7926745852086272*pi) q[38];
U1q(1.97883289272628*pi,0.78645693022104*pi) q[39];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[1],q[19];
RZZ(0.5*pi) q[2],q[17];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[28],q[4];
RZZ(0.5*pi) q[5],q[34];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[39],q[7];
RZZ(0.5*pi) q[9],q[29];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[15],q[13];
RZZ(0.5*pi) q[37],q[14];
RZZ(0.5*pi) q[36],q[18];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[21],q[25];
RZZ(0.5*pi) q[24],q[32];
RZZ(0.5*pi) q[31],q[26];
RZZ(0.5*pi) q[38],q[30];
RZZ(0.5*pi) q[33],q[35];
U1q(0.834176059904628*pi,0.8564664428802597*pi) q[0];
U1q(1.17087154408203*pi,0.25385226553817875*pi) q[1];
U1q(0.510106985581201*pi,0.8329035190857321*pi) q[2];
U1q(3.266460557250395*pi,1.9838792209296736*pi) q[3];
U1q(0.432008256596483*pi,1.0882503097584686*pi) q[4];
U1q(0.588489302832921*pi,1.5986297902357753*pi) q[5];
U1q(1.41721672556941*pi,1.8542685471382656*pi) q[6];
U1q(1.46967145677415*pi,0.22658537266701284*pi) q[7];
U1q(3.259774843032516*pi,0.9509127016719301*pi) q[8];
U1q(3.2806443611473233*pi,1.6976361498424382*pi) q[9];
U1q(1.57566514559205*pi,1.0916518376813595*pi) q[10];
U1q(3.449267732419716*pi,0.25813776335557836*pi) q[11];
U1q(1.62064825654289*pi,0.7855813403389798*pi) q[12];
U1q(0.554901280062903*pi,1.5166502470124894*pi) q[13];
U1q(1.60202570670049*pi,0.890328390274628*pi) q[14];
U1q(1.47735385403098*pi,1.7436179144073485*pi) q[15];
U1q(0.388710924050774*pi,1.6155779294669435*pi) q[16];
U1q(0.618070392989472*pi,0.9936441703378742*pi) q[17];
U1q(0.632006200556176*pi,0.97301887906346*pi) q[18];
U1q(0.211127184170971*pi,1.889009267068318*pi) q[19];
U1q(0.221643032864053*pi,1.4869904801537022*pi) q[20];
U1q(1.42222813933015*pi,0.7060337188681776*pi) q[21];
U1q(3.317350348329762*pi,1.6686749635424096*pi) q[22];
U1q(1.77851006227042*pi,1.4207801939210727*pi) q[23];
U1q(0.249274521551873*pi,0.04498783077214785*pi) q[24];
U1q(0.791941224950558*pi,1.1036326253329372*pi) q[25];
U1q(1.5672992390569*pi,1.6598104269762262*pi) q[26];
U1q(0.53619643664084*pi,0.83206190542787*pi) q[27];
U1q(0.264446222403785*pi,1.1262296679138597*pi) q[28];
U1q(1.70905992226146*pi,1.1607444227739547*pi) q[29];
U1q(0.665650641437571*pi,0.6310090064953604*pi) q[30];
U1q(0.76068376713844*pi,1.737930015189157*pi) q[31];
U1q(0.167200125929291*pi,0.6936838746866023*pi) q[32];
U1q(1.55532540773395*pi,0.8903137221728148*pi) q[33];
U1q(1.76117466940366*pi,0.8932502967154861*pi) q[34];
U1q(0.226136433281298*pi,0.6624721813299248*pi) q[35];
U1q(0.363844696989492*pi,0.8657098929085407*pi) q[36];
U1q(3.618081743026925*pi,0.29673238408816216*pi) q[37];
U1q(0.806809129744473*pi,1.556879365806207*pi) q[38];
U1q(0.471010684800511*pi,0.1812336744028098*pi) q[39];
rz(3.1435335571197403*pi) q[0];
rz(3.7461477344618213*pi) q[1];
rz(3.167096480914268*pi) q[2];
rz(2.0161207790703264*pi) q[3];
rz(2.9117496902415314*pi) q[4];
rz(0.4013702097642246*pi) q[5];
rz(0.1457314528617344*pi) q[6];
rz(1.7734146273329872*pi) q[7];
rz(1.04908729832807*pi) q[8];
rz(2.302363850157562*pi) q[9];
rz(0.9083481623186405*pi) q[10];
rz(3.7418622366444216*pi) q[11];
rz(3.2144186596610202*pi) q[12];
rz(0.4833497529875107*pi) q[13];
rz(3.109671609725372*pi) q[14];
rz(0.2563820855926515*pi) q[15];
rz(2.3844220705330565*pi) q[16];
rz(1.0063558296621258*pi) q[17];
rz(3.02698112093654*pi) q[18];
rz(0.110990732931682*pi) q[19];
rz(2.513009519846298*pi) q[20];
rz(3.2939662811318224*pi) q[21];
rz(2.3313250364575904*pi) q[22];
rz(0.5792198060789273*pi) q[23];
rz(1.9550121692278521*pi) q[24];
rz(2.896367374667063*pi) q[25];
rz(0.34018957302377384*pi) q[26];
rz(3.16793809457213*pi) q[27];
rz(0.8737703320861403*pi) q[28];
rz(0.8392555772260453*pi) q[29];
rz(1.3689909935046396*pi) q[30];
rz(2.262069984810843*pi) q[31];
rz(3.3063161253133977*pi) q[32];
rz(3.109686277827185*pi) q[33];
rz(1.106749703284514*pi) q[34];
rz(1.3375278186700752*pi) q[35];
rz(1.1342901070914593*pi) q[36];
rz(3.703267615911838*pi) q[37];
rz(2.443120634193793*pi) q[38];
rz(1.8187663255971902*pi) q[39];
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