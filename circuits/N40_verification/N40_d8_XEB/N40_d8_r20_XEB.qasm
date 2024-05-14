OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.454721361954855*pi,1.673418094551878*pi) q[0];
U1q(0.684898541927069*pi,1.13036626154242*pi) q[1];
U1q(0.230714995793467*pi,0.912050761625899*pi) q[2];
U1q(0.694763358870073*pi,0.61233135998875*pi) q[3];
U1q(0.605554863898448*pi,1.24243476699906*pi) q[4];
U1q(0.293436014931483*pi,1.3791544587119842*pi) q[5];
U1q(0.285904010658088*pi,0.285624071819908*pi) q[6];
U1q(0.519540420310789*pi,0.61674150302023*pi) q[7];
U1q(0.500929561091886*pi,1.34778220616416*pi) q[8];
U1q(0.180282621582197*pi,0.7381811450096001*pi) q[9];
U1q(0.560627472298749*pi,0.236594662150431*pi) q[10];
U1q(0.180052868498697*pi,1.9821347927985482*pi) q[11];
U1q(0.587625017551927*pi,0.243076689561976*pi) q[12];
U1q(0.301076713047755*pi,1.700399519519404*pi) q[13];
U1q(0.643378443149952*pi,1.32806770743373*pi) q[14];
U1q(0.488901406681425*pi,0.229122709586139*pi) q[15];
U1q(0.274770602557913*pi,0.396907275329285*pi) q[16];
U1q(0.912632662209287*pi,1.898774318487819*pi) q[17];
U1q(0.461510583603059*pi,0.478002121753844*pi) q[18];
U1q(0.699134917256224*pi,1.252448228788474*pi) q[19];
U1q(0.34198711533875*pi,1.7393225597603519*pi) q[20];
U1q(0.607066556975526*pi,0.480960262140869*pi) q[21];
U1q(0.310680016629906*pi,0.48070371816659*pi) q[22];
U1q(0.432543721441956*pi,1.2668627452033752*pi) q[23];
U1q(0.617434636194963*pi,0.288988503096441*pi) q[24];
U1q(0.710584746570439*pi,1.18199239750525*pi) q[25];
U1q(0.402984113416267*pi,0.380491869795132*pi) q[26];
U1q(0.259035268704903*pi,0.6569499709809801*pi) q[27];
U1q(0.620579186351102*pi,0.789536186304971*pi) q[28];
U1q(0.515122402893745*pi,0.312173537508039*pi) q[29];
U1q(0.45242568746494*pi,1.89578319227755*pi) q[30];
U1q(0.635763206381123*pi,0.475890941309994*pi) q[31];
U1q(0.2634495012561*pi,0.0459454416029997*pi) q[32];
U1q(0.100293605993473*pi,0.0389766791749423*pi) q[33];
U1q(0.580029059148539*pi,1.573011828078399*pi) q[34];
U1q(0.731044011233223*pi,1.18059620157976*pi) q[35];
U1q(0.667280964705721*pi,0.32058359338052*pi) q[36];
U1q(0.623607511772816*pi,0.448371013297286*pi) q[37];
U1q(0.480944368365033*pi,0.68326860505453*pi) q[38];
U1q(0.811373737872544*pi,0.546914361323965*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[1],q[23];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[25],q[3];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[8],q[12];
RZZ(0.5*pi) q[35],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[11],q[24];
RZZ(0.5*pi) q[37],q[13];
RZZ(0.5*pi) q[14],q[33];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[32],q[18];
RZZ(0.5*pi) q[19],q[20];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[31],q[26];
RZZ(0.5*pi) q[38],q[28];
RZZ(0.5*pi) q[39],q[36];
U1q(0.77987083904262*pi,1.65843291110806*pi) q[0];
U1q(0.788856736219887*pi,0.445933446041106*pi) q[1];
U1q(0.282138974571924*pi,1.86924239229944*pi) q[2];
U1q(0.827273795313051*pi,0.05161414134825004*pi) q[3];
U1q(0.599973308652506*pi,0.427384988926824*pi) q[4];
U1q(0.2514616323115*pi,1.9116988272901598*pi) q[5];
U1q(0.447442095080019*pi,0.17724161999286991*pi) q[6];
U1q(0.347821195889215*pi,0.26710996144211996*pi) q[7];
U1q(0.41690575541339*pi,1.46102847876438*pi) q[8];
U1q(0.179292151835743*pi,1.6614608704553202*pi) q[9];
U1q(0.927963020254446*pi,0.52212243555084*pi) q[10];
U1q(0.412231796219901*pi,1.69579035024049*pi) q[11];
U1q(0.71674415680898*pi,1.98979996017514*pi) q[12];
U1q(0.465060959029703*pi,0.25420376816268986*pi) q[13];
U1q(0.595008451000384*pi,0.593795092007545*pi) q[14];
U1q(0.231086096595236*pi,0.2521180816875299*pi) q[15];
U1q(0.127616658798887*pi,1.2741332702749602*pi) q[16];
U1q(0.614285177141771*pi,1.9121165481243398*pi) q[17];
U1q(0.571798036196681*pi,0.5622271515167101*pi) q[18];
U1q(0.817831525283907*pi,0.20420730471104998*pi) q[19];
U1q(0.171120911089489*pi,0.02354389234315013*pi) q[20];
U1q(0.0793652631213219*pi,0.7733691059352601*pi) q[21];
U1q(0.529477180412992*pi,0.03503541294455004*pi) q[22];
U1q(0.375672794946456*pi,1.7986421118948002*pi) q[23];
U1q(0.526330909001096*pi,0.9833095184216001*pi) q[24];
U1q(0.604022736286645*pi,0.0952734609853958*pi) q[25];
U1q(0.494736805779639*pi,1.230932497453336*pi) q[26];
U1q(0.532856026362098*pi,0.8274610827212898*pi) q[27];
U1q(0.694182013650622*pi,0.12790848340805994*pi) q[28];
U1q(0.34458211064836*pi,1.3089980582457201*pi) q[29];
U1q(0.653819257507854*pi,0.64958763514983*pi) q[30];
U1q(0.778502932680938*pi,1.642853799717312*pi) q[31];
U1q(0.701197263822617*pi,1.3804446096510699*pi) q[32];
U1q(0.647732498490058*pi,0.5776352153338502*pi) q[33];
U1q(0.765986312665861*pi,0.6282031676102902*pi) q[34];
U1q(0.792318343646822*pi,0.73064601304257*pi) q[35];
U1q(0.584777191482916*pi,1.9252334974145602*pi) q[36];
U1q(0.507748187212082*pi,1.9392012539884602*pi) q[37];
U1q(0.246556287989939*pi,1.085359440929411*pi) q[38];
U1q(0.631827477489845*pi,0.209498125657423*pi) q[39];
RZZ(0.5*pi) q[19],q[0];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[20],q[3];
RZZ(0.5*pi) q[23],q[4];
RZZ(0.5*pi) q[37],q[5];
RZZ(0.5*pi) q[27],q[7];
RZZ(0.5*pi) q[31],q[8];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[24],q[10];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[13],q[12];
RZZ(0.5*pi) q[28],q[14];
RZZ(0.5*pi) q[36],q[15];
RZZ(0.5*pi) q[16],q[33];
RZZ(0.5*pi) q[25],q[18];
RZZ(0.5*pi) q[35],q[21];
RZZ(0.5*pi) q[22],q[34];
RZZ(0.5*pi) q[26],q[29];
RZZ(0.5*pi) q[30],q[39];
U1q(0.305923344582982*pi,0.5829570347562703*pi) q[0];
U1q(0.767020876404544*pi,0.26815973340129995*pi) q[1];
U1q(0.0938344157903267*pi,0.3308225913345999*pi) q[2];
U1q(0.300755661101169*pi,0.09716891412560003*pi) q[3];
U1q(0.652433262174557*pi,0.20553772961565997*pi) q[4];
U1q(0.713893050365061*pi,0.23908662938208014*pi) q[5];
U1q(0.689751170579526*pi,1.0946560482550796*pi) q[6];
U1q(0.271852635045809*pi,0.9628127477947301*pi) q[7];
U1q(0.751623761310796*pi,0.007244432911660104*pi) q[8];
U1q(0.258152692472453*pi,0.30930743134245997*pi) q[9];
U1q(0.754834497531055*pi,1.9071440949615104*pi) q[10];
U1q(0.843700321585697*pi,1.4060019858818298*pi) q[11];
U1q(0.0327612952966543*pi,0.9063311718935703*pi) q[12];
U1q(0.619706049260405*pi,0.21301777670248967*pi) q[13];
U1q(0.730700100379023*pi,1.260805523691009*pi) q[14];
U1q(0.782727175693727*pi,1.1693940481784804*pi) q[15];
U1q(0.275713743054219*pi,1.2459983262628196*pi) q[16];
U1q(0.774014128792221*pi,0.4654423407433801*pi) q[17];
U1q(0.566265515592118*pi,1.5321381151577196*pi) q[18];
U1q(0.264169540630162*pi,0.63897371959282*pi) q[19];
U1q(0.470920901421716*pi,0.36289506266449*pi) q[20];
U1q(0.911221602819065*pi,0.9049394337877601*pi) q[21];
U1q(0.748188442327401*pi,0.99195058233963*pi) q[22];
U1q(0.888146009687891*pi,1.6850277850757003*pi) q[23];
U1q(0.608463155792868*pi,1.78621516152944*pi) q[24];
U1q(0.509683363517474*pi,1.2034940039018198*pi) q[25];
U1q(0.288785351739718*pi,1.04036111316164*pi) q[26];
U1q(0.280356989868079*pi,0.5342925350304597*pi) q[27];
U1q(0.463275150461985*pi,1.0320364869224*pi) q[28];
U1q(0.399992620505687*pi,1.4255313331926702*pi) q[29];
U1q(0.504739002611245*pi,1.3122193808300704*pi) q[30];
U1q(0.618538430636318*pi,0.3288690936659999*pi) q[31];
U1q(0.53665724095254*pi,0.25951951643455007*pi) q[32];
U1q(0.616244388357476*pi,0.6936340045603302*pi) q[33];
U1q(0.613765296749711*pi,0.2405807492852503*pi) q[34];
U1q(0.331413434221856*pi,0.2615619467573298*pi) q[35];
U1q(0.172318609153124*pi,0.93300882885947*pi) q[36];
U1q(0.471199528391165*pi,0.45176286186532*pi) q[37];
U1q(0.753142489966*pi,1.0778850311442398*pi) q[38];
U1q(0.681009188953631*pi,0.06077007893042996*pi) q[39];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[4],q[15];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[8],q[33];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[11],q[23];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[21],q[14];
RZZ(0.5*pi) q[17],q[29];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[38],q[24];
RZZ(0.5*pi) q[30],q[25];
RZZ(0.5*pi) q[31],q[27];
RZZ(0.5*pi) q[32],q[34];
U1q(0.425748693653684*pi,1.2800006800453403*pi) q[0];
U1q(0.350347463800427*pi,0.01563125474338989*pi) q[1];
U1q(0.458139523196252*pi,1.72561642170451*pi) q[2];
U1q(0.887068150954426*pi,1.9803922219575192*pi) q[3];
U1q(0.309732650020797*pi,1.8588126374443998*pi) q[4];
U1q(0.173411678962501*pi,1.6681418149790304*pi) q[5];
U1q(0.763630489586002*pi,1.6478535939677696*pi) q[6];
U1q(0.400684748008066*pi,1.0461090613692*pi) q[7];
U1q(0.306366220591347*pi,1.516496649494*pi) q[8];
U1q(0.414978248135215*pi,0.6517704874939598*pi) q[9];
U1q(0.776853812030044*pi,1.7796705951896303*pi) q[10];
U1q(0.677552965038639*pi,1.9003445423130696*pi) q[11];
U1q(0.378698368894309*pi,0.94251480989137*pi) q[12];
U1q(0.389596673332623*pi,0.3078541098079999*pi) q[13];
U1q(0.297418670933221*pi,1.2155632533426601*pi) q[14];
U1q(0.310169078136129*pi,1.0215113903161104*pi) q[15];
U1q(0.717568670946956*pi,0.5023217573997201*pi) q[16];
U1q(0.692458233299145*pi,1.2740892882571*pi) q[17];
U1q(0.282540143509371*pi,0.13691252815073973*pi) q[18];
U1q(0.399022724062694*pi,1.1605414989756095*pi) q[19];
U1q(0.636903906652639*pi,1.3969448152502704*pi) q[20];
U1q(0.471553359314492*pi,0.7655399428258498*pi) q[21];
U1q(0.468852487022746*pi,0.47447280553672*pi) q[22];
U1q(0.730791847185015*pi,0.004399692486139806*pi) q[23];
U1q(0.485399943484338*pi,0.4988617964858202*pi) q[24];
U1q(0.902819597935586*pi,0.83482168621443*pi) q[25];
U1q(0.488192522420556*pi,1.9227444314117*pi) q[26];
U1q(0.607810114534315*pi,1.8023598285514701*pi) q[27];
U1q(0.647274331410923*pi,0.47732171996779993*pi) q[28];
U1q(0.452533647884513*pi,1.3683539333171009*pi) q[29];
U1q(0.473806452059546*pi,1.8760151925040196*pi) q[30];
U1q(0.798995619376696*pi,1.8156143126727198*pi) q[31];
U1q(0.226497728940937*pi,1.4135149006666201*pi) q[32];
U1q(0.34353780065138*pi,1.8777150954816202*pi) q[33];
U1q(0.163056375816988*pi,0.5339696004520498*pi) q[34];
U1q(0.486497592248309*pi,0.6632062000910599*pi) q[35];
U1q(0.26855777042412*pi,1.6988372802799407*pi) q[36];
U1q(0.60669873801256*pi,1.4261151136627905*pi) q[37];
U1q(0.432335633990493*pi,1.53799109671532*pi) q[38];
U1q(0.568786565693858*pi,1.6305427281678302*pi) q[39];
RZZ(0.5*pi) q[0],q[15];
RZZ(0.5*pi) q[1],q[14];
RZZ(0.5*pi) q[22],q[2];
RZZ(0.5*pi) q[11],q[3];
RZZ(0.5*pi) q[38],q[4];
RZZ(0.5*pi) q[5],q[33];
RZZ(0.5*pi) q[8],q[6];
RZZ(0.5*pi) q[13],q[7];
RZZ(0.5*pi) q[9],q[29];
RZZ(0.5*pi) q[26],q[10];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[31],q[20];
RZZ(0.5*pi) q[21],q[28];
RZZ(0.5*pi) q[23],q[24];
RZZ(0.5*pi) q[35],q[27];
RZZ(0.5*pi) q[30],q[32];
RZZ(0.5*pi) q[39],q[34];
U1q(0.597915754469124*pi,1.0853849076561009*pi) q[0];
U1q(0.267507667063058*pi,0.9427657147877895*pi) q[1];
U1q(0.501975770750172*pi,0.22732015747289935*pi) q[2];
U1q(0.605852606664037*pi,1.6641775331250006*pi) q[3];
U1q(0.450281257161089*pi,0.8783517267872201*pi) q[4];
U1q(0.362623139459529*pi,0.8593927918831294*pi) q[5];
U1q(0.341526999640565*pi,1.5899104685712793*pi) q[6];
U1q(0.702096579986082*pi,0.8533542383032806*pi) q[7];
U1q(0.261135082219171*pi,1.73407053980248*pi) q[8];
U1q(0.622124079895381*pi,1.18562308965533*pi) q[9];
U1q(0.331412043431708*pi,1.2479512952614993*pi) q[10];
U1q(0.489159756667059*pi,0.22317248389278*pi) q[11];
U1q(0.269652206241774*pi,0.8964201477317495*pi) q[12];
U1q(0.144212707900258*pi,1.2028307837493006*pi) q[13];
U1q(0.731975819300304*pi,0.55320861014977*pi) q[14];
U1q(0.563823608215233*pi,0.0487812640100902*pi) q[15];
U1q(0.798745050788458*pi,0.5574751420340007*pi) q[16];
U1q(0.235974473342668*pi,0.7932862051604701*pi) q[17];
U1q(0.851672442737266*pi,0.4062052158743601*pi) q[18];
U1q(0.670020434039773*pi,1.6433314506453005*pi) q[19];
U1q(0.39462874396751*pi,0.2482618976392601*pi) q[20];
U1q(0.626206062612233*pi,1.2502452507500603*pi) q[21];
U1q(0.350600544348753*pi,0.8302711881859999*pi) q[22];
U1q(0.418496634032627*pi,1.2625261060555797*pi) q[23];
U1q(0.732957289709282*pi,1.73570662502644*pi) q[24];
U1q(0.648169035368406*pi,1.0140730866037702*pi) q[25];
U1q(0.146908232208798*pi,0.3817918326653391*pi) q[26];
U1q(0.644206368433258*pi,0.8202859433491598*pi) q[27];
U1q(0.644589277824674*pi,1.7280495014190702*pi) q[28];
U1q(0.759529586325319*pi,0.5906847111850997*pi) q[29];
U1q(0.460341842144838*pi,0.6091539366744705*pi) q[30];
U1q(0.372908049039757*pi,0.18626049376272036*pi) q[31];
U1q(0.771325074022167*pi,1.1486285352591397*pi) q[32];
U1q(0.201567230088697*pi,1.2755984958466602*pi) q[33];
U1q(0.375263786517421*pi,1.3381491238212995*pi) q[34];
U1q(0.43985403699762*pi,0.5971542996126002*pi) q[35];
U1q(0.526286109516008*pi,0.8466951867572092*pi) q[36];
U1q(0.140181206248041*pi,1.2198935592540998*pi) q[37];
U1q(0.271733259660215*pi,1.7786154808949792*pi) q[38];
U1q(0.790760769052839*pi,0.46113848429061033*pi) q[39];
RZZ(0.5*pi) q[35],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[4],q[36];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[15];
RZZ(0.5*pi) q[16],q[7];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[19],q[10];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[32],q[17];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[28],q[22];
RZZ(0.5*pi) q[24],q[34];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[38],q[39];
U1q(0.0513813682567715*pi,1.5392481686991992*pi) q[0];
U1q(0.755544180055558*pi,0.7796248097864993*pi) q[1];
U1q(0.296880391752101*pi,1.3294774819910007*pi) q[2];
U1q(0.33335090841287*pi,1.2361784559561002*pi) q[3];
U1q(0.480244483560555*pi,0.68158367229851*pi) q[4];
U1q(0.145421562400693*pi,1.5627260927061997*pi) q[5];
U1q(0.262324569105917*pi,0.05334973497500073*pi) q[6];
U1q(0.122536151820084*pi,1.8148737855362995*pi) q[7];
U1q(0.453342136707939*pi,0.34026849337169995*pi) q[8];
U1q(0.821420485351499*pi,0.13779684481296073*pi) q[9];
U1q(0.778047699039346*pi,0.6673361651788206*pi) q[10];
U1q(0.442678925573692*pi,0.6120945293841702*pi) q[11];
U1q(0.062410914489433*pi,1.8077543612388993*pi) q[12];
U1q(0.617671363183199*pi,1.3906304454470995*pi) q[13];
U1q(0.52713381920387*pi,0.5975002660879696*pi) q[14];
U1q(0.886299736043652*pi,0.16612418307735943*pi) q[15];
U1q(0.489947662899656*pi,0.11859243407869968*pi) q[16];
U1q(0.596171092957748*pi,1.7955159637412006*pi) q[17];
U1q(0.596811782599978*pi,0.6057811218694296*pi) q[18];
U1q(0.65286821118944*pi,1.1191843274177007*pi) q[19];
U1q(0.372096621584308*pi,0.1822321117048702*pi) q[20];
U1q(0.205401936558888*pi,1.8633594092350894*pi) q[21];
U1q(0.827646709841427*pi,1.9513364780914202*pi) q[22];
U1q(0.464941246334939*pi,1.6359320183414994*pi) q[23];
U1q(0.553067015292325*pi,1.4421127891029997*pi) q[24];
U1q(0.637975645001588*pi,1.4258670045663795*pi) q[25];
U1q(0.349623096960351*pi,1.0341299230278995*pi) q[26];
U1q(0.563342794816183*pi,0.30307434500443*pi) q[27];
U1q(0.483464468507241*pi,0.21765031551190983*pi) q[28];
U1q(0.299911112527209*pi,1.3634188918729002*pi) q[29];
U1q(0.5760893181982*pi,0.03190638671570056*pi) q[30];
U1q(0.416588731719425*pi,1.5871875402912003*pi) q[31];
U1q(0.457887464132616*pi,0.5198127541424*pi) q[32];
U1q(0.31082475798494*pi,1.7439011551185004*pi) q[33];
U1q(0.403824844484065*pi,0.16798960244859984*pi) q[34];
U1q(0.667702427094592*pi,1.5751994861537995*pi) q[35];
U1q(0.545129489950549*pi,1.4409429365561994*pi) q[36];
U1q(0.253527321526184*pi,0.6399327958202008*pi) q[37];
U1q(0.526645781617133*pi,0.030634369295599484*pi) q[38];
U1q(0.383596188203711*pi,1.45922597415119*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[1],q[9];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[26],q[3];
RZZ(0.5*pi) q[4],q[27];
RZZ(0.5*pi) q[5],q[7];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[11],q[29];
RZZ(0.5*pi) q[33],q[12];
RZZ(0.5*pi) q[24],q[14];
RZZ(0.5*pi) q[17],q[15];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[18],q[19];
RZZ(0.5*pi) q[30],q[20];
RZZ(0.5*pi) q[21],q[23];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[25],q[32];
RZZ(0.5*pi) q[28],q[39];
RZZ(0.5*pi) q[31],q[34];
U1q(0.101826177532256*pi,1.4737261445986007*pi) q[0];
U1q(0.636283845836589*pi,0.6243861576668*pi) q[1];
U1q(0.743694083535214*pi,0.18835282294170064*pi) q[2];
U1q(0.217393347089769*pi,0.3892903215137995*pi) q[3];
U1q(0.575503139741641*pi,0.15410643703576987*pi) q[4];
U1q(0.34246343416313*pi,0.9929404671174993*pi) q[5];
U1q(0.217204843133773*pi,0.4692135041014005*pi) q[6];
U1q(0.437664565514542*pi,0.01080249529359989*pi) q[7];
U1q(0.549120783394569*pi,0.6656026354626405*pi) q[8];
U1q(0.522470949137415*pi,1.5323244561670997*pi) q[9];
U1q(0.185839498891442*pi,0.7256370378140993*pi) q[10];
U1q(0.376727403153726*pi,0.017449977835809705*pi) q[11];
U1q(0.580732358226233*pi,1.3416400317452002*pi) q[12];
U1q(0.356890645676135*pi,0.7384646940724*pi) q[13];
U1q(0.111547887425871*pi,1.2167906907072599*pi) q[14];
U1q(0.433132581313316*pi,1.0596936031537005*pi) q[15];
U1q(0.474513353853391*pi,0.23677522531959916*pi) q[16];
U1q(0.518821184026952*pi,1.3591542151491005*pi) q[17];
U1q(0.771366794579065*pi,1.2776281134079408*pi) q[18];
U1q(0.67984083965598*pi,1.9311983781820992*pi) q[19];
U1q(0.243350036488148*pi,1.2266208755338006*pi) q[20];
U1q(0.274723640414986*pi,1.2743639586238*pi) q[21];
U1q(0.112647968488244*pi,0.4313078905865009*pi) q[22];
U1q(0.263575669309482*pi,0.7560568837716009*pi) q[23];
U1q(0.70949622953643*pi,1.1103654812441999*pi) q[24];
U1q(0.474892289288115*pi,0.5043331796467001*pi) q[25];
U1q(0.61348935558982*pi,0.9420800536165999*pi) q[26];
U1q(0.584024767800378*pi,1.1543051795807795*pi) q[27];
U1q(0.532690395920058*pi,1.9112500893560007*pi) q[28];
U1q(0.275625533426352*pi,0.6311034226417007*pi) q[29];
U1q(0.427460336776968*pi,1.8263363452305992*pi) q[30];
U1q(0.180118417856347*pi,1.658002791566*pi) q[31];
U1q(0.231079083846571*pi,0.7013224576020702*pi) q[32];
U1q(0.493120121776458*pi,0.2854751296267999*pi) q[33];
U1q(0.513107296754721*pi,1.2042213790510008*pi) q[34];
U1q(0.431507236961703*pi,1.6160935847548998*pi) q[35];
U1q(0.660690871595924*pi,0.6189895859683006*pi) q[36];
U1q(0.754173765154091*pi,0.3782325705229006*pi) q[37];
U1q(0.570195042526164*pi,1.6001680672221*pi) q[38];
U1q(0.663730162360701*pi,0.6239294059230005*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[24],q[3];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[32],q[6];
RZZ(0.5*pi) q[14],q[7];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[23],q[10];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[13],q[39];
RZZ(0.5*pi) q[31],q[15];
RZZ(0.5*pi) q[25],q[16];
RZZ(0.5*pi) q[17],q[27];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[19],q[34];
RZZ(0.5*pi) q[21],q[36];
RZZ(0.5*pi) q[22],q[33];
RZZ(0.5*pi) q[37],q[28];
U1q(0.410126987791749*pi,1.1491010664725003*pi) q[0];
U1q(0.414405506725708*pi,1.3851878798392008*pi) q[1];
U1q(0.662476225740263*pi,0.6592740591780988*pi) q[2];
U1q(0.807696836555348*pi,1.6625487297581003*pi) q[3];
U1q(0.381624725943015*pi,0.13274811707150036*pi) q[4];
U1q(0.581268358382609*pi,0.5139299989962005*pi) q[5];
U1q(0.238132933218164*pi,0.371988944004201*pi) q[6];
U1q(0.566354442285391*pi,0.5687615738632008*pi) q[7];
U1q(0.593474211135361*pi,0.5786801659324006*pi) q[8];
U1q(0.615084593714555*pi,0.2935216175775004*pi) q[9];
U1q(0.567330489910479*pi,0.28545257327600027*pi) q[10];
U1q(0.361212964007092*pi,0.8629108704414996*pi) q[11];
U1q(0.633034140902469*pi,0.7496519644113988*pi) q[12];
U1q(0.719963682719498*pi,1.6503998649312983*pi) q[13];
U1q(0.532443030907004*pi,1.8340941521205991*pi) q[14];
U1q(0.415016218658358*pi,0.13116981656069981*pi) q[15];
U1q(0.517732397583261*pi,1.5983163337933988*pi) q[16];
U1q(0.787211119818477*pi,1.0107914540530984*pi) q[17];
U1q(0.629938354925719*pi,0.7266287301554009*pi) q[18];
U1q(0.340059221675317*pi,0.2951570760992013*pi) q[19];
U1q(0.617458345815623*pi,0.8148744461974005*pi) q[20];
U1q(0.160329826738662*pi,0.1397732234176008*pi) q[21];
U1q(0.320284187796136*pi,0.7789410877311997*pi) q[22];
U1q(0.443138674042432*pi,0.3645965079586997*pi) q[23];
U1q(0.469641057874274*pi,0.9468230788863998*pi) q[24];
U1q(0.384336684526401*pi,1.5821321464948*pi) q[25];
U1q(0.498337651091585*pi,0.6820056516339008*pi) q[26];
U1q(0.585684580853518*pi,1.6068081500405*pi) q[27];
U1q(0.342452284817455*pi,0.8714684346689996*pi) q[28];
U1q(0.452532331656033*pi,0.06011234924649855*pi) q[29];
U1q(0.49511788206935*pi,0.46114631046739873*pi) q[30];
U1q(0.467907773848682*pi,1.3464134102895997*pi) q[31];
U1q(0.77700294911856*pi,0.7439113426475696*pi) q[32];
U1q(0.426627540856733*pi,1.6929338153701003*pi) q[33];
U1q(0.606263529367957*pi,1.5746867005915988*pi) q[34];
U1q(0.534277590649942*pi,1.9141178279334987*pi) q[35];
U1q(0.790761583790499*pi,0.24570170868980057*pi) q[36];
U1q(0.697348722601229*pi,1.2057571207946012*pi) q[37];
U1q(0.368098462320423*pi,0.7105545555404014*pi) q[38];
U1q(0.704664457272861*pi,1.7040893918840005*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[1],q[18];
RZZ(0.5*pi) q[2],q[10];
RZZ(0.5*pi) q[38],q[3];
RZZ(0.5*pi) q[20],q[4];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[39],q[6];
RZZ(0.5*pi) q[34],q[7];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[19],q[12];
RZZ(0.5*pi) q[17],q[13];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[33],q[15];
RZZ(0.5*pi) q[32],q[16];
RZZ(0.5*pi) q[21],q[29];
RZZ(0.5*pi) q[31],q[22];
RZZ(0.5*pi) q[35],q[24];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[37],q[27];
U1q(0.678287324656129*pi,1.6625423866878002*pi) q[0];
U1q(0.43537686530218*pi,0.5615348759667*pi) q[1];
U1q(0.424955784320028*pi,1.3500320764407014*pi) q[2];
U1q(0.324159068101209*pi,1.1112161103717995*pi) q[3];
U1q(0.787946662155141*pi,1.2921391604714003*pi) q[4];
U1q(0.873939299122343*pi,1.6600307390413995*pi) q[5];
U1q(0.498787782919266*pi,0.2901606589453998*pi) q[6];
U1q(0.441781576520196*pi,0.7623250727551998*pi) q[7];
U1q(0.752866428298483*pi,0.28055958368680045*pi) q[8];
U1q(0.452312514791982*pi,0.5511792332877992*pi) q[9];
U1q(0.349691323765998*pi,1.9163812221993997*pi) q[10];
U1q(0.429389304937922*pi,1.0263621384664994*pi) q[11];
U1q(0.257439122977115*pi,0.6575111436256016*pi) q[12];
U1q(0.258457309694011*pi,0.1191194641594997*pi) q[13];
U1q(0.309356385448043*pi,0.2225985372721997*pi) q[14];
U1q(0.332817392075486*pi,1.7381199301414014*pi) q[15];
U1q(0.696644836083925*pi,0.5548954864685989*pi) q[16];
U1q(0.385499713255018*pi,0.02191950462480108*pi) q[17];
U1q(0.488625400846416*pi,1.4533555636878006*pi) q[18];
U1q(0.466171823003447*pi,0.42867183979949885*pi) q[19];
U1q(0.726997325716626*pi,0.38556642319730017*pi) q[20];
U1q(0.630918964912254*pi,0.4618055160708998*pi) q[21];
U1q(0.0713752564914252*pi,1.6099921971146998*pi) q[22];
U1q(0.18536563520094*pi,1.9184650748135006*pi) q[23];
U1q(0.55789531465414*pi,0.47324299407499915*pi) q[24];
U1q(0.460712344315863*pi,0.20768250775099872*pi) q[25];
U1q(0.728009392530429*pi,0.6542697925060992*pi) q[26];
U1q(0.368129945846298*pi,1.0819064450690004*pi) q[27];
U1q(0.509496757661594*pi,0.08555723977850072*pi) q[28];
U1q(0.469501443867477*pi,1.4896801716184989*pi) q[29];
U1q(0.50845397155292*pi,1.9567312104077992*pi) q[30];
U1q(0.350347430987367*pi,1.1072289411087013*pi) q[31];
U1q(0.321288663990648*pi,0.20923078860499977*pi) q[32];
U1q(0.298905523431035*pi,0.2089074652769014*pi) q[33];
U1q(0.372751586871392*pi,1.6030510150980994*pi) q[34];
U1q(0.246757869023004*pi,1.0580401714796004*pi) q[35];
U1q(0.756301933129246*pi,1.6860350069075984*pi) q[36];
U1q(0.568455755856693*pi,1.6672054113588004*pi) q[37];
U1q(0.656721297428518*pi,1.5843785139296003*pi) q[38];
U1q(0.146896940893259*pi,1.9147904447917004*pi) q[39];
rz(0.6431655670120016*pi) q[0];
rz(2.2362820374859*pi) q[1];
rz(3.1280606444452985*pi) q[2];
rz(0.27946908454600106*pi) q[3];
rz(0.20417700198420086*pi) q[4];
rz(1.2775254205761009*pi) q[5];
rz(0.16751444046670017*pi) q[6];
rz(2.780824805137101*pi) q[7];
rz(0.5483194865240009*pi) q[8];
rz(0.13700517813990132*pi) q[9];
rz(0.5639410013722994*pi) q[10];
rz(3.4397784125490993*pi) q[11];
rz(3.9434283945371007*pi) q[12];
rz(2.465526294333401*pi) q[13];
rz(1.8088216358226*pi) q[14];
rz(0.3253806423127017*pi) q[15];
rz(2.9368457567826987*pi) q[16];
rz(0.3976496277754009*pi) q[17];
rz(2.7177163793140995*pi) q[18];
rz(0.13873797488900053*pi) q[19];
rz(0.6548469182972987*pi) q[20];
rz(3.5360744818315997*pi) q[21];
rz(3.1855158072036005*pi) q[22];
rz(1.3279105257736994*pi) q[23];
rz(3.4209803529619993*pi) q[24];
rz(1.0411878349118986*pi) q[25];
rz(1.3969123676551014*pi) q[26];
rz(0.7788118045511006*pi) q[27];
rz(3.1677882493242002*pi) q[28];
rz(3.4893999337683006*pi) q[29];
rz(0.39135468658340145*pi) q[30];
rz(0.12119762662180023*pi) q[31];
rz(3.973842628882*pi) q[32];
rz(1.938435632072899*pi) q[33];
rz(3.0079206711549986*pi) q[34];
rz(3.7649351295731*pi) q[35];
rz(1.5186932939556002*pi) q[36];
rz(1.641887102797*pi) q[37];
rz(2.6302714838842007*pi) q[38];
rz(2.2517522443674007*pi) q[39];
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
