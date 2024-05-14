OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.303990045743359*pi,1.863776705271938*pi) q[0];
U1q(0.681262558627455*pi,1.430779165933167*pi) q[1];
U1q(0.628690852430353*pi,1.840874015440518*pi) q[2];
U1q(0.591361687123232*pi,1.207544234269244*pi) q[3];
U1q(0.222157205923448*pi,0.342182633090998*pi) q[4];
U1q(0.409779371122309*pi,0.9924869548353701*pi) q[5];
U1q(0.751121735531307*pi,1.894304832555064*pi) q[6];
U1q(0.751824982618182*pi,1.162701342857002*pi) q[7];
U1q(0.595848729143822*pi,1.922323184593226*pi) q[8];
U1q(0.38604658657387*pi,0.449793202775826*pi) q[9];
U1q(0.58480712137335*pi,1.761372726324713*pi) q[10];
U1q(0.355907877225038*pi,0.561017773845869*pi) q[11];
U1q(0.334576224743838*pi,0.686631476267106*pi) q[12];
U1q(0.456123733595945*pi,1.668275773618586*pi) q[13];
U1q(0.302494275933348*pi,0.085187610078906*pi) q[14];
U1q(0.365865993373703*pi,1.5098569638899089*pi) q[15];
U1q(0.723179826440926*pi,1.612641385916257*pi) q[16];
U1q(0.849065543088728*pi,1.645588048340587*pi) q[17];
U1q(0.665729329576162*pi,1.780859420505287*pi) q[18];
U1q(0.551305134013638*pi,0.635840728753689*pi) q[19];
U1q(0.854352235863529*pi,0.64533891800305*pi) q[20];
U1q(0.559041469358488*pi,1.51685771247529*pi) q[21];
U1q(0.804722757292315*pi,0.94627414799016*pi) q[22];
U1q(0.207184474837965*pi,0.6965347346667501*pi) q[23];
U1q(0.653902145795896*pi,0.426366800335419*pi) q[24];
U1q(0.738448060297952*pi,1.1954703831796*pi) q[25];
U1q(0.882931001698403*pi,0.458492101065564*pi) q[26];
U1q(0.403866657245067*pi,0.9911761668236101*pi) q[27];
U1q(0.864826028807629*pi,1.871780545993458*pi) q[28];
U1q(0.689510622155906*pi,0.276864637503813*pi) q[29];
U1q(0.526985407353715*pi,0.0605471134791521*pi) q[30];
U1q(0.229965137619566*pi,1.778193529457497*pi) q[31];
U1q(0.715914557601923*pi,1.479881575476381*pi) q[32];
U1q(0.27602890517809*pi,0.782492579048078*pi) q[33];
U1q(0.676478564673934*pi,0.0717763485643941*pi) q[34];
U1q(0.432749184456464*pi,0.59342511880386*pi) q[35];
U1q(0.580342380872514*pi,1.635335701513938*pi) q[36];
U1q(0.436589179047798*pi,0.139431937419199*pi) q[37];
U1q(0.760274076426552*pi,1.987708204930653*pi) q[38];
U1q(0.394640586486202*pi,0.197584158884314*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[9],q[1];
RZZ(0.5*pi) q[2],q[25];
RZZ(0.5*pi) q[34],q[3];
RZZ(0.5*pi) q[30],q[4];
RZZ(0.5*pi) q[33],q[5];
RZZ(0.5*pi) q[6],q[23];
RZZ(0.5*pi) q[36],q[7];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[21],q[12];
RZZ(0.5*pi) q[13],q[35];
RZZ(0.5*pi) q[14],q[32];
RZZ(0.5*pi) q[16],q[15];
RZZ(0.5*pi) q[18],q[39];
RZZ(0.5*pi) q[20],q[31];
RZZ(0.5*pi) q[24],q[29];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[27],q[38];
U1q(0.539367537172039*pi,1.74040496489808*pi) q[0];
U1q(0.513714881696126*pi,0.3977444150132001*pi) q[1];
U1q(0.698053319965698*pi,1.89551320317736*pi) q[2];
U1q(0.464919922363581*pi,1.3237698998372198*pi) q[3];
U1q(0.308981989789558*pi,0.6283632192212698*pi) q[4];
U1q(0.6070890202166*pi,1.27525701822743*pi) q[5];
U1q(0.516619822158486*pi,1.9289326297176403*pi) q[6];
U1q(0.579972970877295*pi,0.6725128677439298*pi) q[7];
U1q(0.926128066517043*pi,1.168088869151*pi) q[8];
U1q(0.722216381625111*pi,1.8212072378633*pi) q[9];
U1q(0.63176719159612*pi,1.9044284796720898*pi) q[10];
U1q(0.644228138704507*pi,1.9492065331310502*pi) q[11];
U1q(0.681516604619251*pi,0.942664032358*pi) q[12];
U1q(0.935736555018464*pi,0.03604946514332008*pi) q[13];
U1q(0.874505568437812*pi,1.71803917551214*pi) q[14];
U1q(0.54979792997093*pi,1.21363611899208*pi) q[15];
U1q(0.57503236976583*pi,0.4626154429425502*pi) q[16];
U1q(0.264961384313405*pi,0.42818371502447006*pi) q[17];
U1q(0.5862285974981*pi,1.3356186106450898*pi) q[18];
U1q(0.551523759426295*pi,0.547166064064141*pi) q[19];
U1q(0.58543290612764*pi,1.416647529628737*pi) q[20];
U1q(0.422377822766307*pi,1.469176859829763*pi) q[21];
U1q(0.404601839155158*pi,1.8984223255225796*pi) q[22];
U1q(0.202666299849185*pi,0.52639207865835*pi) q[23];
U1q(0.440409647131227*pi,1.98439721768578*pi) q[24];
U1q(0.432747292389124*pi,0.8902065026921699*pi) q[25];
U1q(0.525434532416815*pi,0.3248869344809999*pi) q[26];
U1q(0.612149983439176*pi,1.63446820298849*pi) q[27];
U1q(0.314236286353358*pi,1.38270590975098*pi) q[28];
U1q(0.716291934448303*pi,1.65250278914302*pi) q[29];
U1q(0.513937317170279*pi,1.38292137068848*pi) q[30];
U1q(0.268999042810476*pi,0.8235019142190398*pi) q[31];
U1q(0.453193488029602*pi,0.23783943191101997*pi) q[32];
U1q(0.448476190300472*pi,0.004419251282330006*pi) q[33];
U1q(0.463236282765192*pi,1.61162997325789*pi) q[34];
U1q(0.182499799374034*pi,0.7818656204610601*pi) q[35];
U1q(0.479781707726737*pi,1.2807138705036798*pi) q[36];
U1q(0.256149127889689*pi,1.5988441565550202*pi) q[37];
U1q(0.750455672356365*pi,0.9501846656041999*pi) q[38];
U1q(0.574135645628102*pi,0.6256242916235499*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[26],q[2];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[5],q[4];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[29];
RZZ(0.5*pi) q[8],q[38];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[17],q[12];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[14],q[25];
RZZ(0.5*pi) q[30],q[15];
RZZ(0.5*pi) q[16],q[32];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[19],q[23];
RZZ(0.5*pi) q[22],q[28];
RZZ(0.5*pi) q[24],q[39];
RZZ(0.5*pi) q[36],q[27];
U1q(0.516108304267976*pi,1.7130331805857404*pi) q[0];
U1q(0.70803323981081*pi,0.27297483893280994*pi) q[1];
U1q(0.372566146689978*pi,0.44210833177015996*pi) q[2];
U1q(0.549689987133222*pi,0.2453571236244798*pi) q[3];
U1q(0.244946941294571*pi,0.88282693506856*pi) q[4];
U1q(0.514731891010337*pi,0.05809989248689984*pi) q[5];
U1q(0.771816055952146*pi,1.36818146759453*pi) q[6];
U1q(0.392111959855105*pi,1.5955835426122098*pi) q[7];
U1q(0.50541002058395*pi,1.51433429865808*pi) q[8];
U1q(0.758450715674694*pi,0.30712145365242005*pi) q[9];
U1q(0.465114478892888*pi,1.9872128855961302*pi) q[10];
U1q(0.440844275081474*pi,1.1393941067490898*pi) q[11];
U1q(0.419781022512633*pi,0.74534691003727*pi) q[12];
U1q(0.840329421409666*pi,1.4841299288467003*pi) q[13];
U1q(0.575000237073562*pi,0.8979851370494698*pi) q[14];
U1q(0.614047713139835*pi,1.3688282685418596*pi) q[15];
U1q(0.34965959289085*pi,1.5734576400499503*pi) q[16];
U1q(0.757573672705509*pi,0.5638346768042899*pi) q[17];
U1q(0.913558997339175*pi,1.2611514597395397*pi) q[18];
U1q(0.73472385861802*pi,1.149531512919616*pi) q[19];
U1q(0.255164411478827*pi,1.99271902350024*pi) q[20];
U1q(0.800045321874018*pi,0.2540149444208999*pi) q[21];
U1q(0.156855587422921*pi,0.38916139647006*pi) q[22];
U1q(0.74553473445952*pi,0.9014873029991*pi) q[23];
U1q(0.396532788873919*pi,0.10941457666076992*pi) q[24];
U1q(0.521357269317914*pi,0.6148426322097702*pi) q[25];
U1q(0.768283853030529*pi,1.63524192611419*pi) q[26];
U1q(0.668500003178143*pi,0.03234365229883007*pi) q[27];
U1q(0.412826075198544*pi,1.5816152241041799*pi) q[28];
U1q(0.482743156393879*pi,0.5075846977196199*pi) q[29];
U1q(0.308761663522345*pi,1.3265099198017003*pi) q[30];
U1q(0.640460792703486*pi,1.60110553329954*pi) q[31];
U1q(0.578263182341731*pi,0.9215391310024303*pi) q[32];
U1q(0.978811353177555*pi,0.49661658408796017*pi) q[33];
U1q(0.61888538994949*pi,0.6635744228086198*pi) q[34];
U1q(0.714664633673104*pi,1.9222651946556404*pi) q[35];
U1q(0.701857439062022*pi,0.8871581091385803*pi) q[36];
U1q(0.78609342549355*pi,0.9053895449163303*pi) q[37];
U1q(0.337160706730955*pi,0.6960121355629898*pi) q[38];
U1q(0.85044426519711*pi,0.1559834457834599*pi) q[39];
RZZ(0.5*pi) q[0],q[29];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[19],q[2];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[6],q[4];
RZZ(0.5*pi) q[5],q[39];
RZZ(0.5*pi) q[11],q[8];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[10],q[15];
RZZ(0.5*pi) q[26],q[12];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[20],q[24];
RZZ(0.5*pi) q[36],q[22];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[28],q[25];
RZZ(0.5*pi) q[34],q[27];
RZZ(0.5*pi) q[32],q[38];
RZZ(0.5*pi) q[35],q[37];
U1q(0.841221052941525*pi,0.95113954543503*pi) q[0];
U1q(0.546161994237855*pi,0.12577102910853988*pi) q[1];
U1q(0.925505794581256*pi,0.4909036010623602*pi) q[2];
U1q(0.552263633391656*pi,1.7964793195564202*pi) q[3];
U1q(0.789416954194739*pi,1.3474806497843197*pi) q[4];
U1q(0.44320622023876*pi,1.23786482220852*pi) q[5];
U1q(0.0231962510522512*pi,0.34774461935029954*pi) q[6];
U1q(0.17592017350567*pi,1.7519420462893098*pi) q[7];
U1q(0.315108583379803*pi,1.1397994783652896*pi) q[8];
U1q(0.259956514969729*pi,1.3667771245288396*pi) q[9];
U1q(0.790242426952998*pi,0.8964883303229794*pi) q[10];
U1q(0.170900086312405*pi,1.4753799818048696*pi) q[11];
U1q(0.64724797415739*pi,1.4508055377727302*pi) q[12];
U1q(0.724841980219991*pi,1.0880324849541498*pi) q[13];
U1q(0.757689011902189*pi,1.4644827186021496*pi) q[14];
U1q(0.290254960999379*pi,1.6332966951389496*pi) q[15];
U1q(0.168890082405464*pi,1.8020962060649008*pi) q[16];
U1q(0.0654650869888922*pi,1.2288701021112995*pi) q[17];
U1q(0.374392339603512*pi,0.6158421349694496*pi) q[18];
U1q(0.111052384174974*pi,1.0830067618644001*pi) q[19];
U1q(0.311038643601038*pi,0.6023880854453303*pi) q[20];
U1q(0.268218194986236*pi,0.16995893299771003*pi) q[21];
U1q(0.261983173908941*pi,0.7620289251245005*pi) q[22];
U1q(0.444898447548121*pi,1.5421837452161*pi) q[23];
U1q(0.238620737826089*pi,0.5655365871173501*pi) q[24];
U1q(0.477009625529428*pi,0.6806765205460197*pi) q[25];
U1q(0.421732275453272*pi,0.96017635050561*pi) q[26];
U1q(0.369313058494631*pi,0.2417702821083001*pi) q[27];
U1q(0.59835811538143*pi,0.9337293616914604*pi) q[28];
U1q(0.125877811459227*pi,0.4202827219781504*pi) q[29];
U1q(0.519401068422829*pi,1.4264149523639498*pi) q[30];
U1q(0.27282100073657*pi,1.8312552668852202*pi) q[31];
U1q(0.559791105608808*pi,0.04692695439019978*pi) q[32];
U1q(0.542543676091389*pi,1.6688460043684197*pi) q[33];
U1q(0.653929543122091*pi,1.1424501986486497*pi) q[34];
U1q(0.395389898011683*pi,1.2449140215409296*pi) q[35];
U1q(0.895578829700585*pi,0.7880333715648797*pi) q[36];
U1q(0.329831409699967*pi,1.5291769473730294*pi) q[37];
U1q(0.548076407131769*pi,0.40181594948498045*pi) q[38];
U1q(0.785360049080459*pi,0.1415330442382503*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[11],q[1];
RZZ(0.5*pi) q[20],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[30],q[7];
RZZ(0.5*pi) q[37],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[10],q[39];
RZZ(0.5*pi) q[33],q[12];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[16],q[19];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[22],q[29];
RZZ(0.5*pi) q[26],q[25];
RZZ(0.5*pi) q[36],q[31];
RZZ(0.5*pi) q[35],q[32];
U1q(0.741112082579683*pi,0.8254717007839103*pi) q[0];
U1q(0.785460712656344*pi,1.4783243730646003*pi) q[1];
U1q(0.305136894894178*pi,0.5169621911371998*pi) q[2];
U1q(0.184160172205285*pi,0.24874592256814942*pi) q[3];
U1q(0.129778395145072*pi,1.9482892394589992*pi) q[4];
U1q(0.422332050520241*pi,0.9239300737814196*pi) q[5];
U1q(0.301514995503578*pi,1.9238552215681004*pi) q[6];
U1q(0.364576590408104*pi,0.2804376574654004*pi) q[7];
U1q(0.603311988815279*pi,0.8408846371475702*pi) q[8];
U1q(0.299486286311405*pi,1.8471280313930993*pi) q[9];
U1q(0.612182638509921*pi,0.5914734825938996*pi) q[10];
U1q(0.829517787233491*pi,0.86642466603577*pi) q[11];
U1q(0.355489727034528*pi,1.5540305934890704*pi) q[12];
U1q(0.650144513482575*pi,1.8853538405536998*pi) q[13];
U1q(0.806040812408534*pi,0.7795500837663498*pi) q[14];
U1q(0.423616561559236*pi,0.007357554704450209*pi) q[15];
U1q(0.645894548705794*pi,1.098509732678*pi) q[16];
U1q(0.105929541465301*pi,0.9164744583704003*pi) q[17];
U1q(0.380133875148969*pi,0.9068134430752597*pi) q[18];
U1q(0.371686822912972*pi,0.16228121704461973*pi) q[19];
U1q(0.307231667412235*pi,1.2887290572739003*pi) q[20];
U1q(0.622233752338603*pi,1.3620740178233*pi) q[21];
U1q(0.487508768870844*pi,0.9541966750092996*pi) q[22];
U1q(0.153098480536333*pi,0.4855518274284307*pi) q[23];
U1q(0.15407054475094*pi,1.6676366698508094*pi) q[24];
U1q(0.136688535135658*pi,1.7658267349494903*pi) q[25];
U1q(0.2660993541546*pi,0.34291885910676*pi) q[26];
U1q(0.946214807105881*pi,1.5703684303769005*pi) q[27];
U1q(0.687770505134197*pi,0.18975451349445027*pi) q[28];
U1q(0.628712827778548*pi,0.08729631955664008*pi) q[29];
U1q(0.335055634172833*pi,0.6180580556923001*pi) q[30];
U1q(0.659593826282818*pi,0.9045976331352605*pi) q[31];
U1q(0.243352354005345*pi,0.4863531782078896*pi) q[32];
U1q(0.0650784401658376*pi,1.9860320614867994*pi) q[33];
U1q(0.537834837358992*pi,1.4817672183552997*pi) q[34];
U1q(0.337011250832337*pi,0.99635978628109*pi) q[35];
U1q(0.494788514890984*pi,1.8137586466292994*pi) q[36];
U1q(0.746214011776325*pi,0.9529936997410005*pi) q[37];
U1q(0.841067593623828*pi,0.5516144433941008*pi) q[38];
U1q(0.387539915739115*pi,1.6250744108464001*pi) q[39];
RZZ(0.5*pi) q[15],q[0];
RZZ(0.5*pi) q[1],q[25];
RZZ(0.5*pi) q[28],q[2];
RZZ(0.5*pi) q[3],q[9];
RZZ(0.5*pi) q[4],q[31];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[6],q[38];
RZZ(0.5*pi) q[35],q[7];
RZZ(0.5*pi) q[21],q[8];
RZZ(0.5*pi) q[10],q[26];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[27],q[12];
RZZ(0.5*pi) q[13],q[24];
RZZ(0.5*pi) q[14],q[39];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[30],q[29];
RZZ(0.5*pi) q[33],q[36];
U1q(0.441602643777048*pi,0.7410635305565796*pi) q[0];
U1q(0.773126137685722*pi,1.7670486524971007*pi) q[1];
U1q(0.205510790732813*pi,0.8314991211138008*pi) q[2];
U1q(0.95493248430606*pi,1.5596574857146006*pi) q[3];
U1q(0.280478118092092*pi,0.2949400255645003*pi) q[4];
U1q(0.523244423140425*pi,0.35721376860760046*pi) q[5];
U1q(0.412389950455243*pi,1.5920877420603006*pi) q[6];
U1q(0.647495388136206*pi,0.6426942751260007*pi) q[7];
U1q(0.576433138669012*pi,0.9241101264086105*pi) q[8];
U1q(0.360739773550743*pi,1.7605414836350999*pi) q[9];
U1q(0.39171395100016*pi,1.7445053873404*pi) q[10];
U1q(0.86342057813416*pi,1.5715777879955493*pi) q[11];
U1q(0.469875360810882*pi,1.2484298120721*pi) q[12];
U1q(0.824070263571079*pi,0.08886847558980016*pi) q[13];
U1q(0.363100875834335*pi,0.6660733744291001*pi) q[14];
U1q(0.728059382675821*pi,0.03961224516070061*pi) q[15];
U1q(0.458744230606058*pi,1.8790449989582996*pi) q[16];
U1q(0.791130848963846*pi,1.1619289513816007*pi) q[17];
U1q(0.13547727421845*pi,1.3359162042366002*pi) q[18];
U1q(0.401128340354675*pi,0.8639154153502293*pi) q[19];
U1q(0.73247474592196*pi,0.5389527420493607*pi) q[20];
U1q(0.651123759407643*pi,0.7301586027071991*pi) q[21];
U1q(0.589788502582381*pi,1.6870617190259996*pi) q[22];
U1q(0.511118076966724*pi,1.2839039161850998*pi) q[23];
U1q(0.826215114625413*pi,1.4721409782434005*pi) q[24];
U1q(0.580235938138368*pi,0.9126486324568699*pi) q[25];
U1q(0.621861733247986*pi,1.6037284028748005*pi) q[26];
U1q(0.434690501418214*pi,0.0861593940821006*pi) q[27];
U1q(0.70516029651595*pi,1.3610478184675205*pi) q[28];
U1q(0.7999137465864*pi,1.5072277047277005*pi) q[29];
U1q(0.517621647616891*pi,0.021485837028299315*pi) q[30];
U1q(0.449540555846348*pi,1.6239897326129*pi) q[31];
U1q(0.488300406185834*pi,0.48337560320799966*pi) q[32];
U1q(0.710938540685702*pi,1.4731011570024997*pi) q[33];
U1q(0.263404864494885*pi,1.9762305669596003*pi) q[34];
U1q(0.441765691077561*pi,0.24227691983850086*pi) q[35];
U1q(0.718803195278617*pi,1.7324532202790994*pi) q[36];
U1q(0.674427952333317*pi,0.15212482543669914*pi) q[37];
U1q(0.85567868051951*pi,0.4885165033656005*pi) q[38];
U1q(0.823036981351665*pi,1.8977514660357002*pi) q[39];
RZZ(0.5*pi) q[26],q[0];
RZZ(0.5*pi) q[35],q[1];
RZZ(0.5*pi) q[2],q[29];
RZZ(0.5*pi) q[3],q[17];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[31];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[9],q[23];
RZZ(0.5*pi) q[10],q[37];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[28],q[12];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[21],q[39];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[33],q[24];
RZZ(0.5*pi) q[36],q[32];
RZZ(0.5*pi) q[34],q[38];
U1q(0.601529452605878*pi,1.7324339202233503*pi) q[0];
U1q(0.57449212562497*pi,0.9902673901110006*pi) q[1];
U1q(0.482271381505652*pi,1.7935584510324993*pi) q[2];
U1q(0.664564351862829*pi,1.6048477099274*pi) q[3];
U1q(0.88193567286919*pi,1.1706544874885996*pi) q[4];
U1q(0.647868483151814*pi,1.9410105235292008*pi) q[5];
U1q(0.59862440693629*pi,0.3780764703389998*pi) q[6];
U1q(0.803598760504161*pi,0.7055167821363*pi) q[7];
U1q(0.888890998393157*pi,1.3715335156711*pi) q[8];
U1q(0.528889167543335*pi,0.8741654574989006*pi) q[9];
U1q(0.433821489424309*pi,1.8826604674840013*pi) q[10];
U1q(0.759795563593983*pi,1.2638805301352*pi) q[11];
U1q(0.573834410854213*pi,1.9910974674073003*pi) q[12];
U1q(0.638694101615523*pi,1.8770187605039013*pi) q[13];
U1q(0.62483858765053*pi,0.5806166091685991*pi) q[14];
U1q(0.278907776465201*pi,1.4076150099502005*pi) q[15];
U1q(0.383864033587496*pi,1.2565658361057999*pi) q[16];
U1q(0.662759731442541*pi,0.6153061404206994*pi) q[17];
U1q(0.71147901396806*pi,0.26928044555680053*pi) q[18];
U1q(0.752809263898096*pi,1.4814109615095*pi) q[19];
U1q(0.652750250313648*pi,0.2627286991973996*pi) q[20];
U1q(0.404738530450349*pi,0.2976948030245996*pi) q[21];
U1q(0.489205680405309*pi,0.8562800884379005*pi) q[22];
U1q(0.203791721759287*pi,0.8994505611962005*pi) q[23];
U1q(0.839320890189767*pi,1.6177306594609*pi) q[24];
U1q(0.409043941187992*pi,1.0356023827718008*pi) q[25];
U1q(0.275259829044594*pi,0.4139591293427003*pi) q[26];
U1q(0.604464682985798*pi,0.3013207016518997*pi) q[27];
U1q(0.829556655886715*pi,0.9234940979273993*pi) q[28];
U1q(0.491407412916201*pi,0.46133579600880026*pi) q[29];
U1q(0.117188951036212*pi,0.8043980947118001*pi) q[30];
U1q(0.411545479991897*pi,1.4462176021078008*pi) q[31];
U1q(0.338066884513526*pi,0.8760321998079998*pi) q[32];
U1q(0.303232321545802*pi,1.4275877240934989*pi) q[33];
U1q(0.355094637701875*pi,1.7198479502539001*pi) q[34];
U1q(0.796099153411888*pi,0.3412266024116999*pi) q[35];
U1q(0.123300142216402*pi,0.8042447387562994*pi) q[36];
U1q(0.406080475170818*pi,1.3870553705642*pi) q[37];
U1q(0.114835661940941*pi,0.5659706491171992*pi) q[38];
U1q(0.279431878618569*pi,0.2676646230356994*pi) q[39];
RZZ(0.5*pi) q[33],q[0];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[6],q[2];
RZZ(0.5*pi) q[24],q[3];
RZZ(0.5*pi) q[4],q[27];
RZZ(0.5*pi) q[5],q[29];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[20],q[8];
RZZ(0.5*pi) q[22],q[9];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[11],q[38];
RZZ(0.5*pi) q[14],q[12];
RZZ(0.5*pi) q[16],q[13];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[18],q[25];
RZZ(0.5*pi) q[36],q[21];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[30],q[31];
RZZ(0.5*pi) q[34],q[37];
U1q(0.611515354437034*pi,1.7952685339097005*pi) q[0];
U1q(0.555773141842696*pi,1.5451126575391*pi) q[1];
U1q(0.854042629404777*pi,1.1809698446017016*pi) q[2];
U1q(0.402458620318189*pi,1.0616360556577007*pi) q[3];
U1q(0.851215862336751*pi,0.8529641017888991*pi) q[4];
U1q(0.478312765594127*pi,1.8592912192658009*pi) q[5];
U1q(0.320596681697139*pi,0.7907525469907988*pi) q[6];
U1q(0.738656236972631*pi,0.49985853943270087*pi) q[7];
U1q(0.343315707211038*pi,1.4857002833657997*pi) q[8];
U1q(0.655504511998651*pi,0.9091171349424982*pi) q[9];
U1q(0.671014474413042*pi,1.2638989397751992*pi) q[10];
U1q(0.390906910575708*pi,0.5688890639906994*pi) q[11];
U1q(0.546485581645736*pi,1.4415622508774*pi) q[12];
U1q(0.524730343151648*pi,1.4269528323920007*pi) q[13];
U1q(0.502677575539881*pi,0.4453440084039002*pi) q[14];
U1q(0.312451099827579*pi,0.6045897011474999*pi) q[15];
U1q(0.487359962445361*pi,1.6648247316806994*pi) q[16];
U1q(0.899435130408216*pi,0.20283499664880011*pi) q[17];
U1q(0.640750329868587*pi,0.7122341814757007*pi) q[18];
U1q(0.298762299798548*pi,1.7305543672765005*pi) q[19];
U1q(0.211945179948068*pi,0.6940698445005999*pi) q[20];
U1q(0.160761743104623*pi,0.5210745156671006*pi) q[21];
U1q(0.347211922853167*pi,0.2891091691022005*pi) q[22];
U1q(0.718344946252411*pi,0.5620934836432987*pi) q[23];
U1q(0.613280952315781*pi,0.7981603706172997*pi) q[24];
U1q(0.291719466302947*pi,1.4678941265418004*pi) q[25];
U1q(0.258818042625275*pi,1.6578508610661995*pi) q[26];
U1q(0.948817770682103*pi,0.9779881424096004*pi) q[27];
U1q(0.438446699103501*pi,1.1605580573985996*pi) q[28];
U1q(0.554787757789374*pi,0.4530487967921992*pi) q[29];
U1q(0.352725931829943*pi,1.739967227533601*pi) q[30];
U1q(0.661947469963838*pi,0.9775838245601989*pi) q[31];
U1q(0.933699911673702*pi,1.3587673156982003*pi) q[32];
U1q(0.863805006559636*pi,0.8784690282395005*pi) q[33];
U1q(0.592627071865847*pi,0.3732780580593005*pi) q[34];
U1q(0.398393386684907*pi,1.0331329639528004*pi) q[35];
U1q(0.198495499789072*pi,1.5723608543435006*pi) q[36];
U1q(0.349400913257657*pi,1.2158446344571985*pi) q[37];
U1q(0.214982099506725*pi,0.951893665557801*pi) q[38];
U1q(0.3038409764733*pi,1.9985728994186012*pi) q[39];
RZZ(0.5*pi) q[35],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[2],q[38];
RZZ(0.5*pi) q[3],q[27];
RZZ(0.5*pi) q[4],q[29];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[6],q[15];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[36],q[8];
RZZ(0.5*pi) q[10],q[33];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[16],q[12];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[20],q[17];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[21],q[23];
RZZ(0.5*pi) q[22],q[39];
RZZ(0.5*pi) q[24],q[32];
RZZ(0.5*pi) q[30],q[25];
U1q(0.147448901817133*pi,1.6882740514225993*pi) q[0];
U1q(0.657451721728311*pi,1.688539410943001*pi) q[1];
U1q(0.206966835206279*pi,1.6589345762658994*pi) q[2];
U1q(0.0899746039402263*pi,1.1082280596093987*pi) q[3];
U1q(0.567546102665858*pi,0.2379870925924017*pi) q[4];
U1q(0.919606339061075*pi,0.13467655736499928*pi) q[5];
U1q(0.612992263866737*pi,1.5791625413020007*pi) q[6];
U1q(0.296629944263001*pi,1.3750971557837985*pi) q[7];
U1q(0.710279071638788*pi,1.3767155561465998*pi) q[8];
U1q(0.539417666988059*pi,0.14592869996879898*pi) q[9];
U1q(0.590174117088978*pi,1.0268647022548016*pi) q[10];
U1q(0.674390041527543*pi,0.6418555003799007*pi) q[11];
U1q(0.850847856105499*pi,1.1794471242336009*pi) q[12];
U1q(0.688655248370471*pi,0.3009870480665988*pi) q[13];
U1q(0.714322369334051*pi,0.09568437736160007*pi) q[14];
U1q(0.73754305733054*pi,0.6772330464066982*pi) q[15];
U1q(0.963229659795679*pi,0.13711901947900174*pi) q[16];
U1q(0.446760189057392*pi,1.731399935249101*pi) q[17];
U1q(0.491946904829155*pi,1.7220921001907001*pi) q[18];
U1q(0.898454905351195*pi,0.08633260578769963*pi) q[19];
U1q(0.710463522629904*pi,0.5917598908186008*pi) q[20];
U1q(0.427839731753626*pi,1.1810988323272014*pi) q[21];
U1q(0.326165666044752*pi,0.752067134706099*pi) q[22];
U1q(0.247487027349691*pi,1.0970620590451006*pi) q[23];
U1q(0.201596110379554*pi,0.9134797111755013*pi) q[24];
U1q(0.903759921006808*pi,0.5774517616889003*pi) q[25];
U1q(0.715085137871216*pi,1.8042451239455986*pi) q[26];
U1q(0.318116288092403*pi,1.4085493567068*pi) q[27];
U1q(0.604158485721356*pi,1.1194655126837993*pi) q[28];
U1q(0.305532740677691*pi,1.2820853234409988*pi) q[29];
U1q(0.529301943793657*pi,0.9219360501922012*pi) q[30];
U1q(0.628626080843624*pi,0.2122280689481002*pi) q[31];
U1q(0.654742243368319*pi,0.7242438457156002*pi) q[32];
U1q(0.459597350632771*pi,0.6800553176625002*pi) q[33];
U1q(0.469202708643954*pi,0.9411615200459984*pi) q[34];
U1q(0.106359521633637*pi,1.5489992808231001*pi) q[35];
U1q(0.493471102564815*pi,0.9902429273257987*pi) q[36];
U1q(0.936033961758468*pi,0.7796103432057002*pi) q[37];
U1q(0.59275166971172*pi,0.5312985732158992*pi) q[38];
U1q(0.484174444791461*pi,0.7358953965134987*pi) q[39];
RZZ(0.5*pi) q[0],q[4];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[3],q[39];
RZZ(0.5*pi) q[24],q[5];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[34],q[13];
RZZ(0.5*pi) q[17],q[14];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[18],q[38];
RZZ(0.5*pi) q[26],q[19];
RZZ(0.5*pi) q[20],q[33];
RZZ(0.5*pi) q[35],q[22];
RZZ(0.5*pi) q[23],q[32];
RZZ(0.5*pi) q[37],q[25];
RZZ(0.5*pi) q[30],q[27];
RZZ(0.5*pi) q[28],q[29];
U1q(0.850038088903665*pi,1.1039660614585003*pi) q[0];
U1q(0.526636141492009*pi,0.45586256948410053*pi) q[1];
U1q(0.919703548345204*pi,1.6190946903424006*pi) q[2];
U1q(0.824475350200514*pi,1.4812910828343*pi) q[3];
U1q(0.520631631315007*pi,0.8835500342336005*pi) q[4];
U1q(0.52683982004962*pi,1.3069517350233006*pi) q[5];
U1q(0.185685409706572*pi,1.8009036861372003*pi) q[6];
U1q(0.715288202679324*pi,1.5709886246070006*pi) q[7];
U1q(0.221263311093353*pi,0.4834088039468014*pi) q[8];
U1q(0.708028624713868*pi,0.7677088802986987*pi) q[9];
U1q(0.496485678104352*pi,1.2713883757947997*pi) q[10];
U1q(0.24231535065534*pi,0.9194729676907016*pi) q[11];
U1q(0.686126493920244*pi,1.0824128322826994*pi) q[12];
U1q(0.200933823872014*pi,0.3256544326287987*pi) q[13];
U1q(0.461939427379843*pi,1.3322840684827995*pi) q[14];
U1q(0.39736586695723*pi,0.6377715748301007*pi) q[15];
U1q(0.646907448460626*pi,0.3061428156104995*pi) q[16];
U1q(0.213187163875007*pi,0.523397631050301*pi) q[17];
U1q(0.83507564648222*pi,1.9464239255044014*pi) q[18];
U1q(0.586316107550201*pi,0.5648423619539003*pi) q[19];
U1q(0.636293963828524*pi,0.41804147088929966*pi) q[20];
U1q(0.303945576383471*pi,1.0191461900297014*pi) q[21];
U1q(0.717910921466307*pi,1.9879255564375988*pi) q[22];
U1q(0.315479990866756*pi,1.9877001458159*pi) q[23];
U1q(0.241277600614855*pi,0.8192184568612007*pi) q[24];
U1q(0.557317887478316*pi,0.219542417336001*pi) q[25];
U1q(0.657208487764502*pi,1.3763766222745986*pi) q[26];
U1q(0.301367339158002*pi,1.7231008004824986*pi) q[27];
U1q(0.849564064540558*pi,0.07954869308209922*pi) q[28];
U1q(0.508289501530828*pi,0.5733105765262003*pi) q[29];
U1q(0.45503226177164*pi,1.4042351271634992*pi) q[30];
U1q(0.175003809459297*pi,1.8133585563332986*pi) q[31];
U1q(0.436212654144613*pi,1.4326117998960015*pi) q[32];
U1q(0.568602241740157*pi,1.0339251674968004*pi) q[33];
U1q(0.310654056490909*pi,1.3090846242292002*pi) q[34];
U1q(0.809672540614681*pi,1.6945434802583001*pi) q[35];
U1q(0.912298795340963*pi,0.2788616761542997*pi) q[36];
U1q(0.374787628973455*pi,1.4537555948869993*pi) q[37];
U1q(0.337482045913946*pi,1.5854257265838996*pi) q[38];
U1q(0.28454401219337*pi,0.117495234641801*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[3],q[29];
RZZ(0.5*pi) q[18],q[4];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[7],q[11];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[10],q[13];
RZZ(0.5*pi) q[32],q[12];
RZZ(0.5*pi) q[15],q[22];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[26],q[17];
RZZ(0.5*pi) q[20],q[30];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[23],q[38];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[35],q[31];
RZZ(0.5*pi) q[36],q[39];
U1q(0.547529818753393*pi,1.8034083533916991*pi) q[0];
U1q(0.411606274996934*pi,1.6258475250601983*pi) q[1];
U1q(0.329449793510838*pi,0.3005716865434991*pi) q[2];
U1q(0.410772292000477*pi,1.4372372929988018*pi) q[3];
U1q(0.376902233509824*pi,0.5492560967662996*pi) q[4];
U1q(0.235577662766243*pi,0.6845825538642991*pi) q[5];
U1q(0.050274223242188*pi,0.7321206891584993*pi) q[6];
U1q(0.414410220840787*pi,1.8753848006648006*pi) q[7];
U1q(0.16421469536308*pi,0.7532034550544005*pi) q[8];
U1q(0.36386981352369*pi,0.22890607067959934*pi) q[9];
U1q(0.186011506268671*pi,0.12851129074899958*pi) q[10];
U1q(0.308982535583117*pi,0.003441979219701352*pi) q[11];
U1q(0.679955631555211*pi,1.6289068811206988*pi) q[12];
U1q(0.959728710158821*pi,0.9498121151885996*pi) q[13];
U1q(0.841424296758969*pi,0.9697540363177986*pi) q[14];
U1q(0.311971441252265*pi,1.8135461050455*pi) q[15];
U1q(0.530138179910587*pi,0.9299147256010016*pi) q[16];
U1q(0.326801814659051*pi,0.025928007668198916*pi) q[17];
U1q(0.448897414817024*pi,0.1799096184347988*pi) q[18];
U1q(0.625298017541418*pi,0.7187590208239989*pi) q[19];
U1q(0.492255941289082*pi,1.7908311867820998*pi) q[20];
U1q(0.32553582075039*pi,0.47216719093979975*pi) q[21];
U1q(0.655582481558033*pi,0.6769090450493991*pi) q[22];
U1q(0.652417234327717*pi,0.29158099427329986*pi) q[23];
U1q(0.36415095047016*pi,0.3838181157343996*pi) q[24];
U1q(0.0318152412449319*pi,0.8021265272988991*pi) q[25];
U1q(0.626646036459062*pi,0.1593732617773007*pi) q[26];
U1q(0.266454529801283*pi,1.7273710191484*pi) q[27];
U1q(0.285057174242028*pi,0.30805174106490085*pi) q[28];
U1q(0.645534951557883*pi,1.271532643175199*pi) q[29];
U1q(0.225240932127336*pi,0.20242055156110084*pi) q[30];
U1q(0.34745856781862*pi,1.1734448874897012*pi) q[31];
U1q(0.269488791576855*pi,1.6348348933601002*pi) q[32];
U1q(0.720482023847312*pi,0.9536804087355009*pi) q[33];
U1q(0.0580355269762764*pi,1.5822473870581*pi) q[34];
U1q(0.441983542981364*pi,1.7167204292221996*pi) q[35];
U1q(0.711564359337022*pi,0.976415850118201*pi) q[36];
U1q(0.338734289309415*pi,0.6346187953792004*pi) q[37];
U1q(0.39761160063636*pi,0.5393724544763998*pi) q[38];
U1q(0.322510929113441*pi,1.4089854032796012*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[18],q[1];
RZZ(0.5*pi) q[21],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[35],q[4];
RZZ(0.5*pi) q[34],q[5];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[16],q[11];
RZZ(0.5*pi) q[19],q[12];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[37],q[14];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[24],q[23];
RZZ(0.5*pi) q[39],q[25];
RZZ(0.5*pi) q[26],q[29];
RZZ(0.5*pi) q[36],q[28];
RZZ(0.5*pi) q[33],q[31];
U1q(0.385696456687913*pi,0.5863108744154992*pi) q[0];
U1q(0.234093737690496*pi,0.6896141049697988*pi) q[1];
U1q(0.387002749028719*pi,1.7827233149414994*pi) q[2];
U1q(0.409413687562706*pi,0.18910055826010108*pi) q[3];
U1q(0.149242504457191*pi,1.4544180344742017*pi) q[4];
U1q(0.140952647176127*pi,0.39162029760160166*pi) q[5];
U1q(0.953195057287922*pi,0.6363340829903983*pi) q[6];
U1q(0.350150073359531*pi,0.9428536550346998*pi) q[7];
U1q(0.660901424931356*pi,0.2992197046519003*pi) q[8];
U1q(0.675413979829584*pi,0.9877905790422012*pi) q[9];
U1q(0.410209099689449*pi,0.2620645675829998*pi) q[10];
U1q(0.0911912344263905*pi,0.007629414575500704*pi) q[11];
U1q(0.780117042556786*pi,0.020206634535501422*pi) q[12];
U1q(0.79193600354909*pi,1.6157498198029003*pi) q[13];
U1q(0.686989332621952*pi,1.5081516962500991*pi) q[14];
U1q(0.511615683095868*pi,0.9966013876587994*pi) q[15];
U1q(0.609060082018609*pi,1.6872506036533998*pi) q[16];
U1q(0.372985456968785*pi,0.9068621871249007*pi) q[17];
U1q(0.277061683270056*pi,1.4151928176217012*pi) q[18];
U1q(0.302754486390012*pi,0.46306623659580026*pi) q[19];
U1q(0.707382400098465*pi,0.06835379036839839*pi) q[20];
U1q(0.798456445302771*pi,0.3627054782405992*pi) q[21];
U1q(0.797695411554582*pi,0.6114835254965989*pi) q[22];
U1q(0.548404688544545*pi,1.3141965034329992*pi) q[23];
U1q(0.494046533256661*pi,1.4978192579387013*pi) q[24];
U1q(0.424263821277587*pi,1.6328577472303003*pi) q[25];
U1q(0.64539114831938*pi,0.9917059942834001*pi) q[26];
U1q(0.412752514678095*pi,1.8303624219288999*pi) q[27];
U1q(0.48333230293253*pi,1.9155680307829996*pi) q[28];
U1q(0.666503522777883*pi,0.6175875762846985*pi) q[29];
U1q(0.659531559717837*pi,1.8333401036694994*pi) q[30];
U1q(0.376407865351369*pi,0.26990636834679904*pi) q[31];
U1q(0.422078634725575*pi,1.8585877451441988*pi) q[32];
U1q(0.132808403976677*pi,0.5524706795266994*pi) q[33];
U1q(0.702116981108223*pi,1.4736503421116005*pi) q[34];
U1q(0.81651843508543*pi,1.7678626787600003*pi) q[35];
U1q(0.214967751719311*pi,0.6463261207364006*pi) q[36];
U1q(0.801965148884412*pi,0.954273629805801*pi) q[37];
U1q(0.936404505332625*pi,0.37851743978869834*pi) q[38];
U1q(0.619139604156476*pi,0.8188576091820998*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[17],q[1];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[16],q[5];
RZZ(0.5*pi) q[6],q[29];
RZZ(0.5*pi) q[7],q[38];
RZZ(0.5*pi) q[14],q[8];
RZZ(0.5*pi) q[24],q[9];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[11],q[32];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[33],q[15];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[28],q[39];
RZZ(0.5*pi) q[34],q[36];
U1q(0.153196581578943*pi,0.6200493682745005*pi) q[0];
U1q(0.601740438754845*pi,1.0046190538662998*pi) q[1];
U1q(0.853934253506323*pi,1.7748324449138977*pi) q[2];
U1q(0.555419741295967*pi,0.12788179928330123*pi) q[3];
U1q(0.77680918530454*pi,0.8308204934679999*pi) q[4];
U1q(0.0632719213008257*pi,1.3217171361289992*pi) q[5];
U1q(0.302276087975817*pi,0.33209378033990333*pi) q[6];
U1q(0.390614601177618*pi,0.1955413392624017*pi) q[7];
U1q(0.203277463347042*pi,1.5692091235235992*pi) q[8];
U1q(0.646010214841687*pi,0.6481817800020018*pi) q[9];
U1q(0.559076160954238*pi,1.5415554665911984*pi) q[10];
U1q(0.638207182021468*pi,0.9478581820056*pi) q[11];
U1q(0.506982191924056*pi,1.6337984525493*pi) q[12];
U1q(0.531463612168063*pi,1.1838554131470005*pi) q[13];
U1q(0.164791016289933*pi,1.2142985640283008*pi) q[14];
U1q(0.425930936375437*pi,1.3541047523086007*pi) q[15];
U1q(0.618968258781499*pi,0.5652269421676017*pi) q[16];
U1q(0.80821705038768*pi,1.4279935347326997*pi) q[17];
U1q(0.178278890877218*pi,0.5778415490491007*pi) q[18];
U1q(0.386734331581526*pi,1.1127159574242995*pi) q[19];
U1q(0.450039870718496*pi,0.9670262054908996*pi) q[20];
U1q(0.202284398969971*pi,0.0983143104289006*pi) q[21];
U1q(0.301316848750428*pi,1.4042531160008984*pi) q[22];
U1q(0.270345435346526*pi,0.9175315016185017*pi) q[23];
U1q(0.600788100804441*pi,0.8375589783440986*pi) q[24];
U1q(0.440004027858278*pi,0.6719373418735017*pi) q[25];
U1q(0.790605685245806*pi,0.5682270420868001*pi) q[26];
U1q(0.390012013697976*pi,1.1027397394515006*pi) q[27];
U1q(0.778748212064957*pi,1.4265074111212996*pi) q[28];
U1q(0.634244408089624*pi,0.16283412297449829*pi) q[29];
U1q(0.340345521551259*pi,0.6789926767804992*pi) q[30];
U1q(0.270797268315186*pi,1.1489113965525988*pi) q[31];
U1q(0.473689194113928*pi,0.08944155228600081*pi) q[32];
U1q(0.297483994683032*pi,0.4190396876389002*pi) q[33];
U1q(0.915947444932129*pi,1.0181898090702006*pi) q[34];
U1q(0.0459637804282913*pi,0.4848538031643983*pi) q[35];
U1q(0.302895995716252*pi,1.5984352062823*pi) q[36];
U1q(0.691997350320133*pi,1.9759127500297993*pi) q[37];
U1q(0.621672377392834*pi,0.27664440301100157*pi) q[38];
U1q(0.402692334975455*pi,0.7833396268985027*pi) q[39];
rz(3.3627487600854984*pi) q[0];
rz(1.7387974796066992*pi) q[1];
rz(0.370606068623097*pi) q[2];
rz(3.837202784517501*pi) q[3];
rz(0.8880676762594*pi) q[4];
rz(0.8305517421248005*pi) q[5];
rz(3.7110173340225003*pi) q[6];
rz(2.6043285523761988*pi) q[7];
rz(1.7292849655596996*pi) q[8];
rz(0.8361369202355*pi) q[9];
rz(0.6822030113969006*pi) q[10];
rz(3.242175612432*pi) q[11];
rz(1.508494124640599*pi) q[12];
rz(3.2761851262181985*pi) q[13];
rz(2.023695833449999*pi) q[14];
rz(2.0868109130076*pi) q[15];
rz(3.1432486560498987*pi) q[16];
rz(2.4622664645422994*pi) q[17];
rz(0.21043356097580102*pi) q[18];
rz(2.9612857857833*pi) q[19];
rz(2.788980372831201*pi) q[20];
rz(0.22652814256920095*pi) q[21];
rz(2.917528304516999*pi) q[22];
rz(1.5323616519240986*pi) q[23];
rz(1.9278029533714012*pi) q[24];
rz(3.5873794703793997*pi) q[25];
rz(3.933662663511001*pi) q[26];
rz(2.964761746163301*pi) q[27];
rz(1.9084709689927983*pi) q[28];
rz(0.5706972278280986*pi) q[29];
rz(2.7199489884132007*pi) q[30];
rz(1.8037996424370988*pi) q[31];
rz(2.309837550079699*pi) q[32];
rz(1.9652083439835017*pi) q[33];
rz(3.4863671463246995*pi) q[34];
rz(1.4312217649593997*pi) q[35];
rz(1.5487428546387996*pi) q[36];
rz(3.034804044661101*pi) q[37];
rz(3.1780192163341994*pi) q[38];
rz(1.7819920787724968*pi) q[39];
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
