OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.740080985871626*pi,0.435948800266979*pi) q[0];
U1q(0.829883769404763*pi,1.56913024780033*pi) q[1];
U1q(0.148765929899244*pi,0.136068730443996*pi) q[2];
U1q(0.3914520373195*pi,1.317883228392905*pi) q[3];
U1q(0.909088004835819*pi,0.0743145974692488*pi) q[4];
U1q(0.304407651495237*pi,1.275280042762664*pi) q[5];
U1q(0.742210774976632*pi,1.0743148869439*pi) q[6];
U1q(0.38731323921811*pi,1.453060544413215*pi) q[7];
U1q(0.400721712908003*pi,1.733182148048415*pi) q[8];
U1q(0.461932486686873*pi,1.32701081123134*pi) q[9];
U1q(0.334029838307499*pi,1.212413400713148*pi) q[10];
U1q(0.885534832706127*pi,1.61882378794066*pi) q[11];
U1q(0.561030619216162*pi,1.7259826327904921*pi) q[12];
U1q(0.642822286066378*pi,1.350680777824838*pi) q[13];
U1q(0.717384717665024*pi,1.694263443861791*pi) q[14];
U1q(0.502742705268392*pi,0.598197174794333*pi) q[15];
U1q(0.587438918803051*pi,0.040963904606353*pi) q[16];
U1q(0.831667458009814*pi,1.782862531943366*pi) q[17];
U1q(0.185114604166543*pi,0.7639590473601501*pi) q[18];
U1q(0.469595306017681*pi,1.9630830121851643*pi) q[19];
U1q(0.396626001892993*pi,0.377221944080298*pi) q[20];
U1q(0.939313617753237*pi,1.48013005431899*pi) q[21];
U1q(0.713845206689653*pi,0.275602124188184*pi) q[22];
U1q(0.438835988546509*pi,0.127340092161181*pi) q[23];
U1q(0.694772067051011*pi,1.51768870344616*pi) q[24];
U1q(0.709905890416989*pi,1.0461479118715*pi) q[25];
U1q(0.442280578306018*pi,1.4824562395465382*pi) q[26];
U1q(0.613012306071023*pi,1.650876688945932*pi) q[27];
U1q(0.195768419605802*pi,1.752986717053259*pi) q[28];
U1q(0.550768820477445*pi,1.9438496079689922*pi) q[29];
U1q(0.378150959250812*pi,1.1500522325537421*pi) q[30];
U1q(0.160732592050624*pi,0.492498397286419*pi) q[31];
U1q(0.301054888506475*pi,0.0960740033681207*pi) q[32];
U1q(0.27959519534069*pi,0.884016627231396*pi) q[33];
U1q(0.521989487778991*pi,0.726838786062735*pi) q[34];
U1q(0.314456624177923*pi,0.89634550497553*pi) q[35];
U1q(0.275922966616654*pi,0.237302400579813*pi) q[36];
U1q(0.904096371078088*pi,1.523839262109018*pi) q[37];
U1q(0.419625013134138*pi,1.640236780028151*pi) q[38];
U1q(0.542733834696324*pi,0.83453175279423*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[29],q[2];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[8],q[4];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[15],q[23];
RZZ(0.5*pi) q[32],q[16];
RZZ(0.5*pi) q[34],q[17];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[24],q[26];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[30],q[39];
RZZ(0.5*pi) q[33],q[31];
RZZ(0.5*pi) q[35],q[38];
U1q(0.70998382857356*pi,0.3812877637327401*pi) q[0];
U1q(0.785964174420732*pi,1.624206968307434*pi) q[1];
U1q(0.475316347781583*pi,0.9834735547601501*pi) q[2];
U1q(0.848732079874554*pi,0.20621578362439008*pi) q[3];
U1q(0.686139719013582*pi,0.94159464371108*pi) q[4];
U1q(0.495924501800239*pi,1.7991230439691899*pi) q[5];
U1q(0.187321483766543*pi,0.9793085468423699*pi) q[6];
U1q(0.244627576897519*pi,0.5660770750042601*pi) q[7];
U1q(0.582242114317777*pi,1.28560647545494*pi) q[8];
U1q(0.662789477233573*pi,0.26394117113118*pi) q[9];
U1q(0.274400669300809*pi,0.51637096603503*pi) q[10];
U1q(0.183721315416547*pi,0.8624488619299502*pi) q[11];
U1q(0.500493877124315*pi,1.6624664456978602*pi) q[12];
U1q(0.487276320007308*pi,1.4382608246574602*pi) q[13];
U1q(0.509678302132219*pi,1.2663116514256298*pi) q[14];
U1q(0.456208452702874*pi,1.5055745569719*pi) q[15];
U1q(0.283924236336482*pi,0.4976987154834198*pi) q[16];
U1q(0.304653534068251*pi,0.60997761068514*pi) q[17];
U1q(0.297424819594221*pi,0.6789929301973601*pi) q[18];
U1q(0.283877849244523*pi,1.4911552712578402*pi) q[19];
U1q(0.404336853771666*pi,0.54913820500524*pi) q[20];
U1q(0.599274121377953*pi,1.13619476933416*pi) q[21];
U1q(0.53070517654878*pi,1.71810131942605*pi) q[22];
U1q(0.832016362686545*pi,0.10814534424754996*pi) q[23];
U1q(0.564618663262858*pi,1.3750582048389641*pi) q[24];
U1q(0.798722785093637*pi,1.67917188314957*pi) q[25];
U1q(0.322580974333379*pi,0.599142556011*pi) q[26];
U1q(0.803446156746746*pi,0.6117678638339901*pi) q[27];
U1q(0.408890726353923*pi,1.64679385426061*pi) q[28];
U1q(0.321440325120486*pi,1.1268996492538799*pi) q[29];
U1q(0.160808369296267*pi,0.5993406570930899*pi) q[30];
U1q(0.258242835158983*pi,0.91523517789728*pi) q[31];
U1q(0.389942648525218*pi,0.34119340216672*pi) q[32];
U1q(0.664608382211523*pi,0.002586605990039903*pi) q[33];
U1q(0.726442387659609*pi,0.42273682980672*pi) q[34];
U1q(0.239469633130434*pi,0.9550056747202902*pi) q[35];
U1q(0.615389461781033*pi,0.50271917281218*pi) q[36];
U1q(0.516531419245359*pi,1.85577630920671*pi) q[37];
U1q(0.617053246569817*pi,1.79017448807561*pi) q[38];
U1q(0.693739194994078*pi,1.3189449084726999*pi) q[39];
RZZ(0.5*pi) q[29],q[0];
RZZ(0.5*pi) q[12],q[1];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[3],q[14];
RZZ(0.5*pi) q[4],q[18];
RZZ(0.5*pi) q[22],q[5];
RZZ(0.5*pi) q[9],q[6];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[30],q[10];
RZZ(0.5*pi) q[11],q[26];
RZZ(0.5*pi) q[13],q[37];
RZZ(0.5*pi) q[28],q[15];
RZZ(0.5*pi) q[25],q[17];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[24],q[21];
RZZ(0.5*pi) q[31],q[27];
RZZ(0.5*pi) q[33],q[39];
RZZ(0.5*pi) q[34],q[38];
RZZ(0.5*pi) q[35],q[36];
U1q(0.524945120331252*pi,0.10000720212976022*pi) q[0];
U1q(0.277427767226083*pi,0.9875763189145603*pi) q[1];
U1q(0.608377158378831*pi,0.5696331900039202*pi) q[2];
U1q(0.304078965744285*pi,1.5013956684230196*pi) q[3];
U1q(0.693010031796849*pi,1.1240902624292701*pi) q[4];
U1q(0.552200991924552*pi,1.1410111139994603*pi) q[5];
U1q(0.515352601420178*pi,1.86332671047998*pi) q[6];
U1q(0.491510323085971*pi,0.5939542764361398*pi) q[7];
U1q(0.238820898204466*pi,0.6415079292221204*pi) q[8];
U1q(0.684942954371664*pi,1.582417884487684*pi) q[9];
U1q(0.71250538536599*pi,1.23536734108216*pi) q[10];
U1q(0.627085254001544*pi,0.0053973852919400045*pi) q[11];
U1q(0.18602090746252*pi,0.7792913332079996*pi) q[12];
U1q(0.940925107397709*pi,1.2757474891582996*pi) q[13];
U1q(0.633144753185322*pi,0.48574472762251997*pi) q[14];
U1q(0.190309818457317*pi,1.0615987072363602*pi) q[15];
U1q(0.305186086200989*pi,0.8274501473925104*pi) q[16];
U1q(0.39642717739444*pi,1.4780851259029504*pi) q[17];
U1q(0.376858766554451*pi,1.4704289089471603*pi) q[18];
U1q(0.459211324552728*pi,1.7503293394079096*pi) q[19];
U1q(0.306426529606025*pi,0.13269039030921004*pi) q[20];
U1q(0.320931754377514*pi,1.557466136412012*pi) q[21];
U1q(0.736750406054401*pi,1.0130810796744196*pi) q[22];
U1q(0.547064947325636*pi,0.8927614828463399*pi) q[23];
U1q(0.724912993627764*pi,1.77666023274152*pi) q[24];
U1q(0.366344121472772*pi,0.09578550031646982*pi) q[25];
U1q(0.25213357211323*pi,0.9331734373904403*pi) q[26];
U1q(0.851094112915374*pi,1.7620612956687296*pi) q[27];
U1q(0.506227845250716*pi,0.47985341357581035*pi) q[28];
U1q(0.519637557029234*pi,1.8365426991263796*pi) q[29];
U1q(0.533663595862526*pi,1.7004256138379903*pi) q[30];
U1q(0.915407340985832*pi,0.044874665462260044*pi) q[31];
U1q(0.426090517306394*pi,1.7118594055534802*pi) q[32];
U1q(0.629947796888181*pi,1.2180662573515302*pi) q[33];
U1q(0.92363039940933*pi,0.5394839154716902*pi) q[34];
U1q(0.259655264581159*pi,1.6267279063844304*pi) q[35];
U1q(0.519653459285114*pi,1.2369113294386596*pi) q[36];
U1q(0.494704607726206*pi,0.27114915045621*pi) q[37];
U1q(0.220346476772796*pi,1.6446680030001701*pi) q[38];
U1q(0.52158335492196*pi,0.12412166238422984*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[17],q[1];
RZZ(0.5*pi) q[2],q[6];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[28],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[26],q[10];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[14],q[18];
RZZ(0.5*pi) q[16],q[19];
RZZ(0.5*pi) q[33],q[21];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[24],q[30];
RZZ(0.5*pi) q[37],q[27];
RZZ(0.5*pi) q[29],q[35];
RZZ(0.5*pi) q[39],q[31];
RZZ(0.5*pi) q[32],q[38];
U1q(0.514045718940776*pi,0.16236844947844986*pi) q[0];
U1q(0.632125569565523*pi,0.98534846326314*pi) q[1];
U1q(0.557696584556301*pi,0.7946260999951802*pi) q[2];
U1q(0.149727591852121*pi,0.5248337952008102*pi) q[3];
U1q(0.178624118217077*pi,1.4466079652532997*pi) q[4];
U1q(0.449329878273284*pi,1.4587980361971997*pi) q[5];
U1q(0.973014700661566*pi,0.14113080183805993*pi) q[6];
U1q(0.316558552507841*pi,0.3141757303399597*pi) q[7];
U1q(0.977527940699049*pi,0.9042322738956301*pi) q[8];
U1q(0.510240720277019*pi,0.88043618813536*pi) q[9];
U1q(0.344217517501984*pi,1.6620072489782007*pi) q[10];
U1q(0.340721899844065*pi,1.1352693907511302*pi) q[11];
U1q(0.363666471091738*pi,1.1768102220511398*pi) q[12];
U1q(0.255216990009581*pi,0.5460743009160103*pi) q[13];
U1q(0.902972373272035*pi,1.3425861671127102*pi) q[14];
U1q(0.554070688987566*pi,1.44561596608869*pi) q[15];
U1q(0.671712348465264*pi,0.3365045151314696*pi) q[16];
U1q(0.666418347648214*pi,1.5523150701190005*pi) q[17];
U1q(0.304009128753615*pi,1.9809169018633792*pi) q[18];
U1q(0.591337517409455*pi,1.5517290089394606*pi) q[19];
U1q(0.330916113066384*pi,1.95296465708857*pi) q[20];
U1q(0.18665641860317*pi,1.0617307264230398*pi) q[21];
U1q(0.345327210123244*pi,0.24078281343215036*pi) q[22];
U1q(0.940965645244588*pi,1.9560925385229702*pi) q[23];
U1q(0.55632371418798*pi,1.2708366369597401*pi) q[24];
U1q(0.446292146497124*pi,0.46711225011276003*pi) q[25];
U1q(0.610140042675493*pi,1.5424469824929998*pi) q[26];
U1q(0.942770990402752*pi,1.37805889957116*pi) q[27];
U1q(0.466802936930162*pi,1.9271130745319507*pi) q[28];
U1q(0.596481426842533*pi,1.2430944888079303*pi) q[29];
U1q(0.258029408963128*pi,1.40108937120673*pi) q[30];
U1q(0.502091405138758*pi,0.4155938237528902*pi) q[31];
U1q(0.598534921181368*pi,1.5127040244999908*pi) q[32];
U1q(0.366904391014841*pi,1.4079212900537996*pi) q[33];
U1q(0.324503389516759*pi,0.08470887710340014*pi) q[34];
U1q(0.938987879190266*pi,0.30426891638254006*pi) q[35];
U1q(0.323798032197732*pi,1.3240351163181003*pi) q[36];
U1q(0.0904958536777699*pi,1.0289620376296202*pi) q[37];
U1q(0.475875018105323*pi,1.2577333815392002*pi) q[38];
U1q(0.614081201024573*pi,0.32746674956669963*pi) q[39];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[1],q[38];
RZZ(0.5*pi) q[26],q[2];
RZZ(0.5*pi) q[3],q[34];
RZZ(0.5*pi) q[25],q[4];
RZZ(0.5*pi) q[24],q[5];
RZZ(0.5*pi) q[19],q[6];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[23],q[10];
RZZ(0.5*pi) q[12],q[11];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[17],q[16];
RZZ(0.5*pi) q[31],q[18];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[37],q[21];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[36],q[27];
RZZ(0.5*pi) q[33],q[32];
U1q(0.891329539161351*pi,1.3201618585263795*pi) q[0];
U1q(0.210829097116408*pi,0.08481232785812942*pi) q[1];
U1q(0.416638170764959*pi,0.6555145172522501*pi) q[2];
U1q(0.0728473156697883*pi,1.8897283969671292*pi) q[3];
U1q(0.424679843179343*pi,0.5976333483954503*pi) q[4];
U1q(0.683745995247476*pi,1.7168219700924006*pi) q[5];
U1q(0.628635739387903*pi,0.6782963513131897*pi) q[6];
U1q(0.379240628604981*pi,0.60297844608791*pi) q[7];
U1q(0.641574339936751*pi,0.7894875114500097*pi) q[8];
U1q(0.842277485655044*pi,0.19759945490191022*pi) q[9];
U1q(0.34946952766469*pi,1.8971068723658*pi) q[10];
U1q(0.491818782849576*pi,0.7821567700530903*pi) q[11];
U1q(0.12868368022074*pi,0.3252802368918193*pi) q[12];
U1q(0.374100843796033*pi,1.0859453568630002*pi) q[13];
U1q(0.723212848861873*pi,0.16885239471919*pi) q[14];
U1q(0.271051476578485*pi,0.3809291256157401*pi) q[15];
U1q(0.817810252048969*pi,1.9110842576251006*pi) q[16];
U1q(0.0730292028037064*pi,1.5632658366395997*pi) q[17];
U1q(0.105306987272291*pi,0.5047666950478806*pi) q[18];
U1q(0.556316443342115*pi,0.18407541137822925*pi) q[19];
U1q(0.508627773943242*pi,0.16204512080516054*pi) q[20];
U1q(0.808365963727609*pi,1.9168726648710397*pi) q[21];
U1q(0.651470893577433*pi,0.4092623540509397*pi) q[22];
U1q(0.259788429223199*pi,0.42237689611920004*pi) q[23];
U1q(0.728078931427808*pi,0.6594618341009699*pi) q[24];
U1q(0.456646950051657*pi,1.3137080154377507*pi) q[25];
U1q(0.507673453783166*pi,1.6976470216279704*pi) q[26];
U1q(0.343552510322727*pi,1.9497859416307008*pi) q[27];
U1q(0.584248922053593*pi,1.8383031692985004*pi) q[28];
U1q(0.704871284252051*pi,0.18970779827951922*pi) q[29];
U1q(0.473712499918612*pi,0.6008980847469401*pi) q[30];
U1q(0.186590802571695*pi,0.01841573142086972*pi) q[31];
U1q(0.18946562724609*pi,1.7081149770920998*pi) q[32];
U1q(0.865150092564042*pi,0.15832005995171006*pi) q[33];
U1q(0.473801357432797*pi,1.2116741254564296*pi) q[34];
U1q(0.737884008503845*pi,0.0859189251802901*pi) q[35];
U1q(0.607781143918437*pi,1.4514396631262994*pi) q[36];
U1q(0.314987261592413*pi,1.60327733368345*pi) q[37];
U1q(0.424593928032944*pi,1.2905241992112906*pi) q[38];
U1q(0.137166747929951*pi,1.9623539504291596*pi) q[39];
RZZ(0.5*pi) q[0],q[18];
RZZ(0.5*pi) q[19],q[1];
RZZ(0.5*pi) q[25],q[2];
RZZ(0.5*pi) q[3],q[4];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[14],q[7];
RZZ(0.5*pi) q[37],q[8];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[32],q[12];
RZZ(0.5*pi) q[13],q[22];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[26],q[16];
RZZ(0.5*pi) q[23],q[17];
RZZ(0.5*pi) q[24],q[39];
RZZ(0.5*pi) q[27],q[38];
RZZ(0.5*pi) q[29],q[34];
RZZ(0.5*pi) q[30],q[31];
RZZ(0.5*pi) q[33],q[35];
U1q(0.396505946872996*pi,1.6090691749936*pi) q[0];
U1q(0.230775354745118*pi,1.6695024497256998*pi) q[1];
U1q(0.669781394185057*pi,1.9617483683411994*pi) q[2];
U1q(0.362123119123817*pi,0.7581188550924995*pi) q[3];
U1q(0.873325897708871*pi,1.6629004422481*pi) q[4];
U1q(0.416395433553687*pi,0.2516287148490992*pi) q[5];
U1q(0.693870156368726*pi,1.5243867517908303*pi) q[6];
U1q(0.246609411254758*pi,0.8462341109849003*pi) q[7];
U1q(0.575785673752253*pi,0.11497359097191939*pi) q[8];
U1q(0.876509745995684*pi,0.3930259377797203*pi) q[9];
U1q(0.208825822701095*pi,0.12735717046519923*pi) q[10];
U1q(0.557410848773111*pi,1.5132469347434991*pi) q[11];
U1q(0.0519376489209642*pi,0.5534686833906992*pi) q[12];
U1q(0.0799551325483132*pi,1.8621946668523002*pi) q[13];
U1q(0.198574922708278*pi,1.6404728312118007*pi) q[14];
U1q(0.329366664018147*pi,1.4521462632016*pi) q[15];
U1q(0.688047497446869*pi,1.9754013414867*pi) q[16];
U1q(0.252573073577871*pi,0.6147101525238003*pi) q[17];
U1q(0.364572280927266*pi,1.6629294732250006*pi) q[18];
U1q(0.82096118449003*pi,0.6412317476378995*pi) q[19];
U1q(0.396836044178772*pi,1.9369283065952008*pi) q[20];
U1q(0.139444305367546*pi,1.1612666775099*pi) q[21];
U1q(0.735569479656398*pi,1.0843994729390598*pi) q[22];
U1q(0.419804014589468*pi,0.7414868790386997*pi) q[23];
U1q(0.235957383119731*pi,0.30162042668280087*pi) q[24];
U1q(0.512553023843451*pi,1.2795414163144994*pi) q[25];
U1q(0.30575567544415*pi,1.5539569380930995*pi) q[26];
U1q(0.648164893891868*pi,1.8262803569549*pi) q[27];
U1q(0.538869529726454*pi,1.8697087069865006*pi) q[28];
U1q(0.692657656001024*pi,1.1726826521584996*pi) q[29];
U1q(0.32833056978401*pi,1.7141862419238993*pi) q[30];
U1q(0.72745207893306*pi,0.3097156136724504*pi) q[31];
U1q(0.233291069270173*pi,0.5503715111355998*pi) q[32];
U1q(0.545803803361841*pi,1.5471006196023005*pi) q[33];
U1q(0.862426739477876*pi,1.2554166451783004*pi) q[34];
U1q(0.605335506094247*pi,1.9631417713317703*pi) q[35];
U1q(0.150581628265132*pi,0.0017275768878004527*pi) q[36];
U1q(0.681833275027279*pi,1.9065998768096009*pi) q[37];
U1q(0.91492138653482*pi,0.5722702090788001*pi) q[38];
U1q(0.595802013814937*pi,1.2836858112654994*pi) q[39];
RZZ(0.5*pi) q[2],q[0];
RZZ(0.5*pi) q[39],q[1];
RZZ(0.5*pi) q[3],q[7];
RZZ(0.5*pi) q[37],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[16],q[6];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[9],q[36];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[12],q[30];
RZZ(0.5*pi) q[13],q[18];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[28],q[19];
RZZ(0.5*pi) q[21],q[31];
RZZ(0.5*pi) q[24],q[22];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[35],q[27];
RZZ(0.5*pi) q[33],q[29];
U1q(0.444724028710958*pi,1.2864860900196007*pi) q[0];
U1q(0.57140859828187*pi,0.24622959524210053*pi) q[1];
U1q(0.604444138486605*pi,1.1700362688743997*pi) q[2];
U1q(0.343960796952036*pi,0.05020623750900022*pi) q[3];
U1q(0.834079384296515*pi,1.0941738290492005*pi) q[4];
U1q(0.75082973648041*pi,1.5744172920677002*pi) q[5];
U1q(0.367628529592221*pi,0.26416478950867983*pi) q[6];
U1q(0.205183983378831*pi,1.4353656594899995*pi) q[7];
U1q(0.325917534883514*pi,1.8235627006783997*pi) q[8];
U1q(0.0989557187927195*pi,1.2682618604364002*pi) q[9];
U1q(0.183227124083119*pi,1.8435181901088988*pi) q[10];
U1q(0.296486558025495*pi,1.9292685302613002*pi) q[11];
U1q(0.435862698029427*pi,1.1320298436850997*pi) q[12];
U1q(0.790962509035225*pi,0.8936051069150999*pi) q[13];
U1q(0.268199050113559*pi,0.8570264765749993*pi) q[14];
U1q(0.472491230442687*pi,1.1556306550721995*pi) q[15];
U1q(0.62816124873219*pi,1.7028849371959005*pi) q[16];
U1q(0.251732324127204*pi,1.6252137350880993*pi) q[17];
U1q(0.591471831861626*pi,0.9810873641461999*pi) q[18];
U1q(0.555890667614503*pi,0.8730742979526003*pi) q[19];
U1q(0.32808637797741*pi,1.3319104739445002*pi) q[20];
U1q(0.406850088758034*pi,0.41792348238974064*pi) q[21];
U1q(0.278654390588132*pi,1.8012898517132996*pi) q[22];
U1q(0.209843256958612*pi,1.0734622022004991*pi) q[23];
U1q(0.0723726944251488*pi,1.1700432333226995*pi) q[24];
U1q(0.62714649965779*pi,1.5643133899125985*pi) q[25];
U1q(0.50495965095912*pi,0.5212459206468001*pi) q[26];
U1q(0.396377982112808*pi,0.9509397845554997*pi) q[27];
U1q(0.810617866745161*pi,0.8771206059193997*pi) q[28];
U1q(0.173519879837238*pi,0.24507668385479953*pi) q[29];
U1q(0.430198249971926*pi,1.6299297057784994*pi) q[30];
U1q(0.366627852663021*pi,0.19982433494569918*pi) q[31];
U1q(0.830529537094916*pi,0.5146391959670993*pi) q[32];
U1q(0.688820227634214*pi,0.022611759872699366*pi) q[33];
U1q(0.493021135078828*pi,0.19880651589370046*pi) q[34];
U1q(0.474956460121157*pi,1.5498611199652998*pi) q[35];
U1q(0.43516506165087*pi,1.5740922213551016*pi) q[36];
U1q(0.456392343053684*pi,0.26091641690550027*pi) q[37];
U1q(0.832343493219454*pi,1.8252861493502994*pi) q[38];
U1q(0.628948037831029*pi,0.8079576158555*pi) q[39];
RZZ(0.5*pi) q[27],q[0];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[3],q[29];
RZZ(0.5*pi) q[17],q[4];
RZZ(0.5*pi) q[12],q[5];
RZZ(0.5*pi) q[38],q[6];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[9],q[18];
RZZ(0.5*pi) q[28],q[10];
RZZ(0.5*pi) q[11],q[16];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[14],q[15];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[21],q[26];
RZZ(0.5*pi) q[25],q[22];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[33],q[30];
RZZ(0.5*pi) q[32],q[31];
RZZ(0.5*pi) q[34],q[37];
U1q(0.648555229216505*pi,0.7390036462871006*pi) q[0];
U1q(0.307260805945132*pi,0.3730731583459992*pi) q[1];
U1q(0.853304241860301*pi,0.03756458010970043*pi) q[2];
U1q(0.706533263226548*pi,0.36957865470930074*pi) q[3];
U1q(0.626781479073655*pi,1.6437993869216996*pi) q[4];
U1q(0.336586235332241*pi,1.0951894036256995*pi) q[5];
U1q(0.593409053571505*pi,1.7012899210960999*pi) q[6];
U1q(0.511904645820565*pi,0.6934471398155999*pi) q[7];
U1q(0.601829521996056*pi,0.31858471160989943*pi) q[8];
U1q(0.260927369239086*pi,0.35334638005289953*pi) q[9];
U1q(0.286792914048228*pi,0.1479176093112997*pi) q[10];
U1q(0.0251109023563009*pi,0.8741677284424014*pi) q[11];
U1q(0.72051954487749*pi,1.1754807046324984*pi) q[12];
U1q(0.584814737326272*pi,0.05552404821710155*pi) q[13];
U1q(0.199882478381279*pi,0.7991886340440999*pi) q[14];
U1q(0.368261006582637*pi,1.6879019010745004*pi) q[15];
U1q(0.0517991836962943*pi,0.43361079066309927*pi) q[16];
U1q(0.510265960908322*pi,0.7149522719298993*pi) q[17];
U1q(0.606648199005431*pi,0.9851143016758002*pi) q[18];
U1q(0.367667997728774*pi,0.9599956924920008*pi) q[19];
U1q(0.383345139236866*pi,0.9667152717611991*pi) q[20];
U1q(0.805464039666116*pi,1.1493856053470992*pi) q[21];
U1q(0.293789941716088*pi,1.9810038154892986*pi) q[22];
U1q(0.225593025903801*pi,1.5884153270100008*pi) q[23];
U1q(0.681010045500731*pi,1.5702355435935011*pi) q[24];
U1q(0.103167169192387*pi,0.6034956468472004*pi) q[25];
U1q(0.450284466466527*pi,1.5732633062123007*pi) q[26];
U1q(0.205838084908566*pi,1.4549859775506988*pi) q[27];
U1q(0.548379912041655*pi,1.951525183846801*pi) q[28];
U1q(0.532050790553166*pi,0.9990492991428006*pi) q[29];
U1q(0.416110661040487*pi,1.3316839495574992*pi) q[30];
U1q(0.294517685099586*pi,1.0049564356047007*pi) q[31];
U1q(0.591117427643593*pi,1.1507305213351007*pi) q[32];
U1q(0.609733915459471*pi,0.7022137325143003*pi) q[33];
U1q(0.229819131801991*pi,0.6080424202031001*pi) q[34];
U1q(0.559554594028307*pi,0.07493104444350074*pi) q[35];
U1q(0.32991117527161*pi,0.8598226147916996*pi) q[36];
U1q(0.684809354845873*pi,1.9773806679658996*pi) q[37];
U1q(0.856539131469039*pi,1.2459007125419*pi) q[38];
U1q(0.0644292841017182*pi,0.9061604366136002*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[1],q[18];
RZZ(0.5*pi) q[30],q[2];
RZZ(0.5*pi) q[3],q[38];
RZZ(0.5*pi) q[14],q[4];
RZZ(0.5*pi) q[5],q[26];
RZZ(0.5*pi) q[21],q[6];
RZZ(0.5*pi) q[28],q[7];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[37],q[10];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[12],q[17];
RZZ(0.5*pi) q[13],q[35];
RZZ(0.5*pi) q[23],q[16];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[29],q[32];
U1q(0.427105773180494*pi,1.7553744555827002*pi) q[0];
U1q(0.141017587203362*pi,1.0300855070870014*pi) q[1];
U1q(0.318637328337032*pi,1.9782783818393987*pi) q[2];
U1q(0.790628523725952*pi,1.6296357706352005*pi) q[3];
U1q(0.702092736815715*pi,1.5848133567999003*pi) q[4];
U1q(0.40128104963697*pi,1.5520722533116*pi) q[5];
U1q(0.467641002367092*pi,1.7463812154427991*pi) q[6];
U1q(0.464428769716952*pi,0.12765396281280061*pi) q[7];
U1q(0.0911914086369038*pi,0.8721759995076006*pi) q[8];
U1q(0.713766181231828*pi,1.4994717149383003*pi) q[9];
U1q(0.617890273635425*pi,0.5197207501065009*pi) q[10];
U1q(0.436236386813513*pi,0.9350712721180017*pi) q[11];
U1q(0.547967423409229*pi,0.04989422573950009*pi) q[12];
U1q(0.476039646617425*pi,0.9844720386950989*pi) q[13];
U1q(0.442749764986161*pi,1.1331863327893004*pi) q[14];
U1q(0.692849304070473*pi,0.5451473103091011*pi) q[15];
U1q(0.279497247597912*pi,0.5124604127999*pi) q[16];
U1q(0.784624033525974*pi,1.0582444241173015*pi) q[17];
U1q(0.296346834211712*pi,0.783891725539501*pi) q[18];
U1q(0.152580860977111*pi,1.9076797114086013*pi) q[19];
U1q(0.280753215083753*pi,1.1580327645000992*pi) q[20];
U1q(0.364764082117648*pi,0.46537621636359994*pi) q[21];
U1q(0.278191050896965*pi,0.4591204620091993*pi) q[22];
U1q(0.164289354981948*pi,1.744090407022501*pi) q[23];
U1q(0.53627727990436*pi,1.845706340893301*pi) q[24];
U1q(0.622092435757745*pi,0.5789091097954007*pi) q[25];
U1q(0.626758932261626*pi,0.5645232390772001*pi) q[26];
U1q(0.684418291999345*pi,0.10665284496990068*pi) q[27];
U1q(0.479880617410383*pi,0.2700756492165013*pi) q[28];
U1q(0.306874879952239*pi,0.5969280603115017*pi) q[29];
U1q(0.728634081170608*pi,0.6788336387581992*pi) q[30];
U1q(0.745490682777907*pi,0.9852467462918995*pi) q[31];
U1q(0.176501109145908*pi,0.40388850922299824*pi) q[32];
U1q(0.55378931438823*pi,0.8888865812565001*pi) q[33];
U1q(0.736069107485736*pi,0.6132980503321015*pi) q[34];
U1q(0.475098014093911*pi,1.3003107520606996*pi) q[35];
U1q(0.119732621903523*pi,0.5878281666267995*pi) q[36];
U1q(0.72366936190656*pi,0.202404146456999*pi) q[37];
U1q(0.777854177690693*pi,1.8638657073411995*pi) q[38];
U1q(0.160862489947968*pi,1.0445557811104003*pi) q[39];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[1],q[4];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[24],q[3];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[18];
RZZ(0.5*pi) q[13],q[8];
RZZ(0.5*pi) q[25],q[9];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[12],q[38];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[32],q[17];
RZZ(0.5*pi) q[26],q[19];
RZZ(0.5*pi) q[20],q[31];
RZZ(0.5*pi) q[22],q[23];
RZZ(0.5*pi) q[29],q[28];
RZZ(0.5*pi) q[33],q[37];
RZZ(0.5*pi) q[35],q[39];
U1q(0.479890887503273*pi,0.2607753289529988*pi) q[0];
U1q(0.568603262813984*pi,1.9519621884123985*pi) q[1];
U1q(0.609652423375581*pi,0.31000668376140084*pi) q[2];
U1q(0.357849349127269*pi,1.3072948741170016*pi) q[3];
U1q(0.620382513298384*pi,1.8560033764344013*pi) q[4];
U1q(0.757633469798556*pi,1.5058619247162*pi) q[5];
U1q(0.861230885975644*pi,1.6273007666809995*pi) q[6];
U1q(0.392186809158695*pi,1.7058356794575005*pi) q[7];
U1q(0.881504551316829*pi,1.8147989435400014*pi) q[8];
U1q(0.566116936251421*pi,1.2715370898476994*pi) q[9];
U1q(0.138628582246321*pi,0.4582577283572995*pi) q[10];
U1q(0.48310502910213*pi,1.0035305194796997*pi) q[11];
U1q(0.342668304297535*pi,0.9276231105449*pi) q[12];
U1q(0.618647145753924*pi,0.18921835019850164*pi) q[13];
U1q(0.730660162726056*pi,0.6717494521747014*pi) q[14];
U1q(0.323316779885602*pi,0.5410672042769988*pi) q[15];
U1q(0.417742425563743*pi,0.12274292157229993*pi) q[16];
U1q(0.581901314154703*pi,0.3679766799190993*pi) q[17];
U1q(0.622303248079202*pi,0.546805090190599*pi) q[18];
U1q(0.66784107792602*pi,0.08385781815690052*pi) q[19];
U1q(0.507221138292791*pi,0.8858273427652001*pi) q[20];
U1q(0.82284104192814*pi,1.9631981333939983*pi) q[21];
U1q(0.994176861774092*pi,0.8523917774757983*pi) q[22];
U1q(0.427642384994525*pi,1.389864326565501*pi) q[23];
U1q(0.316726880890988*pi,1.8123345017720993*pi) q[24];
U1q(0.253456722637993*pi,1.8439583035834985*pi) q[25];
U1q(0.825812917942439*pi,0.47152553960980015*pi) q[26];
U1q(0.44492676692722*pi,0.3077605055974999*pi) q[27];
U1q(0.269024787226844*pi,0.3853267498406012*pi) q[28];
U1q(0.150645264386153*pi,1.9344165891885012*pi) q[29];
U1q(0.679067129500051*pi,1.2290765544877011*pi) q[30];
U1q(0.384860400337108*pi,1.1196326493660003*pi) q[31];
U1q(0.279570223943732*pi,1.3690812561439998*pi) q[32];
U1q(0.7875931931993*pi,0.43759174789829913*pi) q[33];
U1q(0.344708188452358*pi,1.1715949448465999*pi) q[34];
U1q(0.636413272411597*pi,0.3970418316840991*pi) q[35];
U1q(0.605993201009181*pi,1.0142302861864998*pi) q[36];
U1q(0.489483806931877*pi,1.0307961301461006*pi) q[37];
U1q(0.489023873191835*pi,1.6953955010310011*pi) q[38];
U1q(0.281306197901062*pi,0.7505672117347011*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[3],q[18];
RZZ(0.5*pi) q[20],q[4];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[16];
RZZ(0.5*pi) q[28],q[8];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[31],q[10];
RZZ(0.5*pi) q[11],q[15];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[29],q[22];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[33],q[26];
RZZ(0.5*pi) q[35],q[30];
RZZ(0.5*pi) q[32],q[36];
RZZ(0.5*pi) q[37],q[39];
U1q(0.907712558119677*pi,1.6201158130465991*pi) q[0];
U1q(0.862505378208931*pi,0.21268700053180112*pi) q[1];
U1q(0.405048311696451*pi,1.8402436301012983*pi) q[2];
U1q(0.324277392609005*pi,0.006861875238598714*pi) q[3];
U1q(0.773820486729905*pi,0.7242215688125988*pi) q[4];
U1q(0.338962916515617*pi,1.2693756386736013*pi) q[5];
U1q(0.314130515082491*pi,1.4184829760671995*pi) q[6];
U1q(0.434731010679919*pi,1.7737493022930018*pi) q[7];
U1q(0.484732010722307*pi,1.458089274307401*pi) q[8];
U1q(0.602335453081473*pi,0.9267878466187014*pi) q[9];
U1q(0.407892687354605*pi,0.2504644808146992*pi) q[10];
U1q(0.54715505877181*pi,0.4439401645936982*pi) q[11];
U1q(0.851800009555398*pi,0.0413583207463013*pi) q[12];
U1q(0.106793498709538*pi,0.4322253980087005*pi) q[13];
U1q(0.671027552815676*pi,1.6296610046210986*pi) q[14];
U1q(0.761442538612712*pi,1.4638285776070994*pi) q[15];
U1q(0.901093032379129*pi,0.12733150064950038*pi) q[16];
U1q(0.523474855029002*pi,1.2500018036881002*pi) q[17];
U1q(0.604985410816845*pi,1.7205647655873015*pi) q[18];
U1q(0.679687647769708*pi,0.40452276372460005*pi) q[19];
U1q(0.60979820380184*pi,0.054016385704400705*pi) q[20];
U1q(0.449249233093874*pi,1.0653656460556*pi) q[21];
U1q(0.194013805734489*pi,1.4502922719138986*pi) q[22];
U1q(0.666472638688833*pi,1.2320675770532006*pi) q[23];
U1q(0.28665812478089*pi,1.3731786441031986*pi) q[24];
U1q(0.386156003919034*pi,1.041102906776299*pi) q[25];
U1q(0.338129479130998*pi,0.7926731907328985*pi) q[26];
U1q(0.230830946840069*pi,0.20729243120250018*pi) q[27];
U1q(0.190006280670546*pi,0.5066706749710015*pi) q[28];
U1q(0.562418229386803*pi,1.5043847994551989*pi) q[29];
U1q(0.593515294633366*pi,1.3846032464063995*pi) q[30];
U1q(0.493233682346179*pi,0.6621459622468002*pi) q[31];
U1q(0.106127768327769*pi,0.21801937607180122*pi) q[32];
U1q(0.89387899134997*pi,0.04127600421210076*pi) q[33];
U1q(0.240396575234576*pi,1.8685605743324984*pi) q[34];
U1q(0.62087220358294*pi,1.216430781813699*pi) q[35];
U1q(0.736188153308119*pi,0.14605933609080068*pi) q[36];
U1q(0.873322343230464*pi,0.31362081812610043*pi) q[37];
U1q(0.412364023824853*pi,1.7420666812898986*pi) q[38];
U1q(0.441881495886915*pi,0.5959513035435009*pi) q[39];
rz(0.10154238378419933*pi) q[0];
rz(2.835773222415799*pi) q[1];
rz(2.8051085376768015*pi) q[2];
rz(1.512860844841299*pi) q[3];
rz(0.058032709479800104*pi) q[4];
rz(0.8728343338535005*pi) q[5];
rz(0.017661576725000572*pi) q[6];
rz(2.8758460972031017*pi) q[7];
rz(2.8723763828391*pi) q[8];
rz(1.7887462728385017*pi) q[9];
rz(2.156501425027301*pi) q[10];
rz(3.2402325860294*pi) q[11];
rz(3.159620211316099*pi) q[12];
rz(1.875074007883601*pi) q[13];
rz(1.6339646487097*pi) q[14];
rz(2.414113801702399*pi) q[15];
rz(3.2231073010018*pi) q[16];
rz(0.013590594348098506*pi) q[17];
rz(2.7105125732050013*pi) q[18];
rz(3.0684937398367005*pi) q[19];
rz(3.2927500896396005*pi) q[20];
rz(2.5353198458702018*pi) q[21];
rz(2.1535385235356017*pi) q[22];
rz(0.19590438549120037*pi) q[23];
rz(2.9171827939306993*pi) q[24];
rz(3.0532476845676015*pi) q[25];
rz(1.4997861513468997*pi) q[26];
rz(2.2714683937877993*pi) q[27];
rz(0.15615681036959828*pi) q[28];
rz(3.530255986941601*pi) q[29];
rz(3.4174592987093*pi) q[30];
rz(2.102070735686201*pi) q[31];
rz(3.617182439281301*pi) q[32];
rz(3.432784835277701*pi) q[33];
rz(0.3120470015356993*pi) q[34];
rz(2.5918853264908996*pi) q[35];
rz(2.4277180302732013*pi) q[36];
rz(2.2435530766682987*pi) q[37];
rz(1.8052932868801008*pi) q[38];
rz(2.5159006702688984*pi) q[39];
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
