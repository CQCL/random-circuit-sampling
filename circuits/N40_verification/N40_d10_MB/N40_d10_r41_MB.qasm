OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.351331760384051*pi,1.638900878231935*pi) q[0];
U1q(0.425218372692789*pi,1.486929608038149*pi) q[1];
U1q(0.872094080189705*pi,1.337966984864531*pi) q[2];
U1q(1.31734672929832*pi,1.6785054083107527*pi) q[3];
U1q(3.5400032420193472*pi,1.281566855926218*pi) q[4];
U1q(1.63375253799421*pi,0.5325354348921806*pi) q[5];
U1q(1.31597401125482*pi,0.03956955178238703*pi) q[6];
U1q(1.54852638690529*pi,0.3152345045158615*pi) q[7];
U1q(0.504558625111548*pi,1.3092373138678601*pi) q[8];
U1q(0.130287964271053*pi,0.92670683302209*pi) q[9];
U1q(0.680033860790906*pi,0.859678346089329*pi) q[10];
U1q(3.449348657341837*pi,1.3641365566482873*pi) q[11];
U1q(0.768270577978058*pi,0.676167516503344*pi) q[12];
U1q(1.4875124075011*pi,0.23823080011358752*pi) q[13];
U1q(0.665077302374204*pi,0.443392978058612*pi) q[14];
U1q(0.550403851364271*pi,0.125165073604399*pi) q[15];
U1q(1.90116561240415*pi,1.60432458117051*pi) q[16];
U1q(1.67912572943047*pi,1.9872548019544782*pi) q[17];
U1q(1.52329536138957*pi,1.4822435276967492*pi) q[18];
U1q(0.608061611433354*pi,0.333672753339118*pi) q[19];
U1q(0.326027764877972*pi,0.98342675578284*pi) q[20];
U1q(0.428400870946059*pi,1.619510814311632*pi) q[21];
U1q(1.17914799199865*pi,1.6375085400429077*pi) q[22];
U1q(1.18013313348489*pi,0.9011450987482847*pi) q[23];
U1q(1.66619379879847*pi,1.3170962497872738*pi) q[24];
U1q(1.61082143414495*pi,1.2692016383771922*pi) q[25];
U1q(1.39122594361999*pi,0.8122128107622527*pi) q[26];
U1q(0.652127259386781*pi,0.98566959843968*pi) q[27];
U1q(0.847685595591805*pi,0.8494375342690801*pi) q[28];
U1q(0.19338455718697*pi,1.271351090812975*pi) q[29];
U1q(0.339648919573013*pi,0.366431785379112*pi) q[30];
U1q(1.91061059603543*pi,0.7404213674124175*pi) q[31];
U1q(1.66604913053658*pi,0.3322130365223938*pi) q[32];
U1q(0.644344578545044*pi,0.084610191630027*pi) q[33];
U1q(0.395264312440536*pi,0.92774188206488*pi) q[34];
U1q(0.373297397448926*pi,0.623687176208185*pi) q[35];
U1q(1.71559672387015*pi,1.330687670632115*pi) q[36];
U1q(0.404232654412166*pi,0.579046957208985*pi) q[37];
U1q(0.509103315640634*pi,1.751674272910675*pi) q[38];
U1q(0.501892039774633*pi,0.912398713155746*pi) q[39];
RZZ(0.5*pi) q[0],q[33];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[2],q[30];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[12],q[4];
RZZ(0.5*pi) q[39],q[6];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[38],q[9];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[11],q[21];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[35],q[14];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[25],q[17];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[26],q[24];
RZZ(0.5*pi) q[29],q[37];
RZZ(0.5*pi) q[31],q[36];
U1q(0.62372037353425*pi,0.01339111396493009*pi) q[0];
U1q(0.56336705033495*pi,0.43130261281610016*pi) q[1];
U1q(0.892040321695846*pi,0.9116491011986101*pi) q[2];
U1q(0.111403997818166*pi,1.5839037842361026*pi) q[3];
U1q(0.208667723948972*pi,1.090546968192038*pi) q[4];
U1q(0.326684842192157*pi,1.6922266258780807*pi) q[5];
U1q(0.833911092643783*pi,0.46006760862587703*pi) q[6];
U1q(0.355340155269384*pi,0.23652313483121112*pi) q[7];
U1q(0.451361473035717*pi,1.5955313310554202*pi) q[8];
U1q(0.158194360468467*pi,1.3780781470875998*pi) q[9];
U1q(0.497602619324248*pi,1.52557686221345*pi) q[10];
U1q(0.322999284190426*pi,1.8701375753159475*pi) q[11];
U1q(0.724150759437483*pi,1.7464773480464881*pi) q[12];
U1q(0.145130859505726*pi,1.5654152761693076*pi) q[13];
U1q(0.489799089772216*pi,0.24882226624025994*pi) q[14];
U1q(0.417726787619472*pi,1.84195667569267*pi) q[15];
U1q(0.338035794128966*pi,0.14552824290416*pi) q[16];
U1q(0.504251380685132*pi,0.4336746672932583*pi) q[17];
U1q(0.283881141903071*pi,1.434783636605359*pi) q[18];
U1q(0.8312596775403*pi,1.81745195282324*pi) q[19];
U1q(0.633482543370446*pi,1.8089553565648302*pi) q[20];
U1q(0.335527639271616*pi,0.41945098519370005*pi) q[21];
U1q(0.867258683318752*pi,0.009614551910637648*pi) q[22];
U1q(0.758431781381686*pi,0.5657056522976498*pi) q[23];
U1q(0.435211804660343*pi,1.5548059936677938*pi) q[24];
U1q(0.362206045546535*pi,0.4222820086688821*pi) q[25];
U1q(0.585040854954182*pi,1.055681095681123*pi) q[26];
U1q(0.23172888278585*pi,1.623503783105655*pi) q[27];
U1q(0.658775207280812*pi,1.82268181880801*pi) q[28];
U1q(0.0948337186957352*pi,1.4014708377726999*pi) q[29];
U1q(0.666732532810237*pi,0.68222998054195*pi) q[30];
U1q(0.58519806383132*pi,0.20726914441746747*pi) q[31];
U1q(0.782032554871343*pi,0.4538871676169238*pi) q[32];
U1q(0.24704256160929*pi,0.54605977971773*pi) q[33];
U1q(0.338901393947434*pi,0.8377597719965999*pi) q[34];
U1q(0.540206160013935*pi,0.27667726935787007*pi) q[35];
U1q(0.469866671181242*pi,0.16715551797708494*pi) q[36];
U1q(0.3270187592466*pi,0.39527394115018*pi) q[37];
U1q(0.900933429825218*pi,1.0763613967838999*pi) q[38];
U1q(0.758824631857165*pi,0.05080765066666992*pi) q[39];
RZZ(0.5*pi) q[23],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[3],q[9];
RZZ(0.5*pi) q[27],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[28],q[8];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[26],q[13];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[16],q[30];
RZZ(0.5*pi) q[38],q[17];
RZZ(0.5*pi) q[19],q[33];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[29],q[39];
RZZ(0.5*pi) q[36],q[32];
U1q(0.733219663082037*pi,0.20699360793412014*pi) q[0];
U1q(0.27716725022013*pi,1.1454498292435904*pi) q[1];
U1q(0.68434042538812*pi,0.84334845299938*pi) q[2];
U1q(0.365111724565316*pi,1.7341399041942722*pi) q[3];
U1q(0.569663551730929*pi,1.0410249199683381*pi) q[4];
U1q(0.20427579258879*pi,1.11690515585431*pi) q[5];
U1q(0.322194021290991*pi,0.8810896745477272*pi) q[6];
U1q(0.75666423333009*pi,0.8882027717136012*pi) q[7];
U1q(0.36872397142155*pi,1.0728185377456096*pi) q[8];
U1q(0.357518076117252*pi,0.43963453063447044*pi) q[9];
U1q(0.54952981561029*pi,1.1223566504281401*pi) q[10];
U1q(0.368535212143584*pi,1.7595948199252671*pi) q[11];
U1q(0.330731901480703*pi,1.2637295345867199*pi) q[12];
U1q(0.322944509663384*pi,0.9626899326579474*pi) q[13];
U1q(0.717713186868677*pi,0.7213102758290701*pi) q[14];
U1q(0.370810771678185*pi,1.5626079922566003*pi) q[15];
U1q(0.605698478459545*pi,1.6710218524266498*pi) q[16];
U1q(0.277209811738086*pi,1.2684817059250477*pi) q[17];
U1q(0.472749671587903*pi,0.25020403543045955*pi) q[18];
U1q(0.934843064994583*pi,1.9958114086266896*pi) q[19];
U1q(0.852120802873663*pi,0.001508089477810337*pi) q[20];
U1q(0.576078998600707*pi,1.9282024326472103*pi) q[21];
U1q(0.664291434821487*pi,0.6377777384838375*pi) q[22];
U1q(0.492452260259793*pi,1.3641092913855648*pi) q[23];
U1q(0.320458122845337*pi,0.8119860070474938*pi) q[24];
U1q(0.449379044796632*pi,1.3453597960809818*pi) q[25];
U1q(0.43191626405559*pi,0.675612096012463*pi) q[26];
U1q(0.196571053005968*pi,0.5654808816664199*pi) q[27];
U1q(0.472564520623743*pi,1.0130591209812598*pi) q[28];
U1q(0.731108204039919*pi,0.6822158340376498*pi) q[29];
U1q(0.445613957470735*pi,1.46378440750861*pi) q[30];
U1q(0.40091530661252*pi,1.4210819568123672*pi) q[31];
U1q(0.659080231447752*pi,1.8276518087904838*pi) q[32];
U1q(0.527401219071496*pi,1.25963246565254*pi) q[33];
U1q(0.530623271815912*pi,1.75031230325801*pi) q[34];
U1q(0.29365352981936*pi,0.57487476258407*pi) q[35];
U1q(0.12316464420995*pi,1.5805558201208347*pi) q[36];
U1q(0.192351609419782*pi,1.9590181448539399*pi) q[37];
U1q(0.612227583343475*pi,1.73863569180254*pi) q[38];
U1q(0.637630003706263*pi,0.44430799202825*pi) q[39];
RZZ(0.5*pi) q[0],q[28];
RZZ(0.5*pi) q[1],q[30];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[38],q[3];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[24],q[5];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[10],q[31];
RZZ(0.5*pi) q[27],q[11];
RZZ(0.5*pi) q[34],q[13];
RZZ(0.5*pi) q[14],q[15];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[36],q[17];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[22],q[32];
RZZ(0.5*pi) q[23],q[25];
U1q(0.305941738869808*pi,1.2446486680658602*pi) q[0];
U1q(0.462604510419231*pi,1.6933798387665995*pi) q[1];
U1q(0.700418126428001*pi,1.4158927245229798*pi) q[2];
U1q(0.389612048814947*pi,0.841872457688253*pi) q[3];
U1q(0.554759427767341*pi,1.1189996029773086*pi) q[4];
U1q(0.285136898699028*pi,0.5155329699384499*pi) q[5];
U1q(0.881295311387744*pi,1.255355319210187*pi) q[6];
U1q(0.426698339056876*pi,1.7084513191313713*pi) q[7];
U1q(0.659328036161088*pi,0.6014507631255501*pi) q[8];
U1q(0.141246994994005*pi,1.5000364501114403*pi) q[9];
U1q(0.795366633463144*pi,1.275396927088*pi) q[10];
U1q(0.629968827376369*pi,1.0225776542855378*pi) q[11];
U1q(0.456641892735916*pi,0.8274895893627603*pi) q[12];
U1q(0.415912272553839*pi,0.8973526345372775*pi) q[13];
U1q(0.754227681409774*pi,0.8152953796529898*pi) q[14];
U1q(0.83520306829262*pi,0.24599031647843006*pi) q[15];
U1q(0.235762661775949*pi,1.9479048861775006*pi) q[16];
U1q(0.470353403241404*pi,1.8661787913897276*pi) q[17];
U1q(0.750995524705666*pi,0.20969652969779862*pi) q[18];
U1q(0.53752951068865*pi,1.0338205670726506*pi) q[19];
U1q(0.158092094805934*pi,1.3927575464288893*pi) q[20];
U1q(0.706794376435368*pi,0.8617114630325702*pi) q[21];
U1q(0.287063445394862*pi,0.017949984144268072*pi) q[22];
U1q(0.350125048010088*pi,0.4858002622178548*pi) q[23];
U1q(0.49842123301892*pi,1.1387882033105834*pi) q[24];
U1q(0.429232439451236*pi,1.0184188177184117*pi) q[25];
U1q(0.7197001677729*pi,1.0954171533158323*pi) q[26];
U1q(0.263903367699081*pi,1.9238031724591007*pi) q[27];
U1q(0.837909537325797*pi,1.8113562854479408*pi) q[28];
U1q(0.382943097558938*pi,1.3013101390860005*pi) q[29];
U1q(0.423671394635641*pi,0.8937631125814303*pi) q[30];
U1q(0.425037547507079*pi,0.9552741839272274*pi) q[31];
U1q(0.637554866024408*pi,0.25014038703774366*pi) q[32];
U1q(0.576295906320182*pi,0.3692401209314098*pi) q[33];
U1q(0.465826529223025*pi,0.14546537940535043*pi) q[34];
U1q(0.498756327695111*pi,1.8513453382042897*pi) q[35];
U1q(0.480934342682386*pi,0.6285676167321546*pi) q[36];
U1q(0.466194365792323*pi,0.7655817232278599*pi) q[37];
U1q(0.411183906268494*pi,1.9000191386538994*pi) q[38];
U1q(0.354559860539868*pi,0.9155386439948892*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[2],q[5];
RZZ(0.5*pi) q[19],q[3];
RZZ(0.5*pi) q[4],q[21];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[7],q[33];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[20];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[32],q[17];
RZZ(0.5*pi) q[22],q[26];
RZZ(0.5*pi) q[35],q[23];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[31],q[27];
U1q(0.342298132846749*pi,0.8027162063551003*pi) q[0];
U1q(0.402498898037675*pi,0.7334967309288007*pi) q[1];
U1q(0.374154584266191*pi,1.0708390295220998*pi) q[2];
U1q(0.356775716702567*pi,1.5658180530768924*pi) q[3];
U1q(0.372192372234564*pi,0.27577220219801823*pi) q[4];
U1q(0.111331468792827*pi,0.4666170507458798*pi) q[5];
U1q(0.359733211042561*pi,1.4232111090307864*pi) q[6];
U1q(0.496955286420695*pi,1.0371825159291124*pi) q[7];
U1q(0.491673242414619*pi,1.8744033442497*pi) q[8];
U1q(0.552897187046822*pi,0.6643686473647499*pi) q[9];
U1q(0.761638322607369*pi,1.5703041563624298*pi) q[10];
U1q(0.300162956684428*pi,1.5374561694160178*pi) q[11];
U1q(0.334236898618689*pi,1.1849747635844796*pi) q[12];
U1q(0.566133258069375*pi,0.8132172528830477*pi) q[13];
U1q(0.664171476336456*pi,1.2744257099877103*pi) q[14];
U1q(0.731927493298356*pi,0.8235630801031002*pi) q[15];
U1q(0.352890505092655*pi,0.37200448661211105*pi) q[16];
U1q(0.603836995954926*pi,0.8419562869204285*pi) q[17];
U1q(0.758919197180287*pi,1.6470429491055398*pi) q[18];
U1q(0.253583434351411*pi,0.1586343967949997*pi) q[19];
U1q(0.39467401499234*pi,0.32916399291969967*pi) q[20];
U1q(0.288417542186316*pi,1.84587636738609*pi) q[21];
U1q(0.425604685009567*pi,1.9046472704877075*pi) q[22];
U1q(0.558602058636447*pi,0.3074783339005345*pi) q[23];
U1q(0.580663844250589*pi,0.4641860448027728*pi) q[24];
U1q(0.156551465923114*pi,0.056814916752591316*pi) q[25];
U1q(0.827804407221073*pi,0.7776775666450533*pi) q[26];
U1q(0.429892569485743*pi,1.9404907962804003*pi) q[27];
U1q(0.43994401993188*pi,1.6314087613700003*pi) q[28];
U1q(0.27351363093205*pi,1.1654127823059994*pi) q[29];
U1q(0.538767057186834*pi,0.5962507661816803*pi) q[30];
U1q(0.430195019390022*pi,1.958130959501588*pi) q[31];
U1q(0.681129508923862*pi,0.34602999199872375*pi) q[32];
U1q(0.806322900412669*pi,0.5853444967017296*pi) q[33];
U1q(0.615976741467286*pi,1.3621979746021804*pi) q[34];
U1q(0.775542305962635*pi,1.7279934784770603*pi) q[35];
U1q(0.524148221696318*pi,0.030919194989284193*pi) q[36];
U1q(0.831030183813989*pi,0.6745413930874102*pi) q[37];
U1q(0.454203064938202*pi,1.5834664467731994*pi) q[38];
U1q(0.736562677891173*pi,0.6818847735488998*pi) q[39];
RZZ(0.5*pi) q[0],q[4];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[28],q[3];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[17];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[22],q[9];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[12],q[27];
RZZ(0.5*pi) q[14],q[30];
RZZ(0.5*pi) q[15],q[33];
RZZ(0.5*pi) q[18],q[24];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[31],q[21];
RZZ(0.5*pi) q[34],q[23];
RZZ(0.5*pi) q[35],q[25];
RZZ(0.5*pi) q[39],q[38];
U1q(0.35256566228067*pi,0.7484463806236992*pi) q[0];
U1q(0.336616145900368*pi,0.8904030646815002*pi) q[1];
U1q(0.570902965551281*pi,0.9965191027996596*pi) q[2];
U1q(0.444494687245438*pi,1.5060887092739517*pi) q[3];
U1q(0.44711094535565*pi,0.6321070177200188*pi) q[4];
U1q(0.955883929155068*pi,1.2681684694919806*pi) q[5];
U1q(0.237066971389136*pi,1.8889690798094865*pi) q[6];
U1q(0.108678865384613*pi,1.7257577871661614*pi) q[7];
U1q(0.964064025726886*pi,0.018036153993499227*pi) q[8];
U1q(0.73990152288906*pi,1.8465856902890998*pi) q[9];
U1q(0.522383913313181*pi,0.007642316808789573*pi) q[10];
U1q(0.057130385445162*pi,1.172234775850887*pi) q[11];
U1q(0.472851296067245*pi,0.7896268162498998*pi) q[12];
U1q(0.369488798234213*pi,1.5457534698882878*pi) q[13];
U1q(0.757307443042404*pi,1.31026354430149*pi) q[14];
U1q(0.13475590568513*pi,0.0412586904586103*pi) q[15];
U1q(0.608003610093094*pi,1.2080318604916105*pi) q[16];
U1q(0.708611846668356*pi,0.23328316180331843*pi) q[17];
U1q(0.705128497379431*pi,1.0939901716383496*pi) q[18];
U1q(0.68813235837659*pi,0.7245615546432997*pi) q[19];
U1q(0.549240895383064*pi,1.2867117163473*pi) q[20];
U1q(0.319317335703153*pi,0.9269443271195001*pi) q[21];
U1q(0.376958833375452*pi,0.3757939892032063*pi) q[22];
U1q(0.273781948656211*pi,0.07802004023048426*pi) q[23];
U1q(0.0841440665244159*pi,0.0004703340405729506*pi) q[24];
U1q(0.0550100483525465*pi,1.9632211373560917*pi) q[25];
U1q(0.490662803631904*pi,0.25965627478345255*pi) q[26];
U1q(0.532250409294444*pi,1.1187254882232*pi) q[27];
U1q(0.219876757204043*pi,0.04263990881559998*pi) q[28];
U1q(0.266737229272407*pi,1.5602814172064008*pi) q[29];
U1q(0.23085730975365*pi,0.9377962163241502*pi) q[30];
U1q(0.142521776633297*pi,0.1365530825336183*pi) q[31];
U1q(0.679728820211309*pi,0.6881556904381734*pi) q[32];
U1q(0.168110083785445*pi,1.1091715976270393*pi) q[33];
U1q(0.268134153072932*pi,1.4700427411204995*pi) q[34];
U1q(0.397137192194134*pi,0.2362847784911004*pi) q[35];
U1q(0.410083783922868*pi,0.6957527999201147*pi) q[36];
U1q(0.109089251035101*pi,1.0669751375157208*pi) q[37];
U1q(0.726461216884225*pi,1.9168796745343997*pi) q[38];
U1q(0.500923799042404*pi,0.4762116290529992*pi) q[39];
rz(0.6377133646674*pi) q[0];
rz(0.46492552558790123*pi) q[1];
rz(1.67510715183648*pi) q[2];
rz(0.1747343695958481*pi) q[3];
rz(0.1584679135753806*pi) q[4];
rz(0.8431850096888205*pi) q[5];
rz(0.08319749467301207*pi) q[6];
rz(3.0271196614719376*pi) q[7];
rz(2.5717679643592*pi) q[8];
rz(1.2840255154755997*pi) q[9];
rz(3.74463853486973*pi) q[10];
rz(3.268374568328614*pi) q[11];
rz(2.7239814523247006*pi) q[12];
rz(3.781517531247612*pi) q[13];
rz(3.4674269203083403*pi) q[14];
rz(3.4628365702512998*pi) q[15];
rz(3.3151512824249885*pi) q[16];
rz(1.0210684525862224*pi) q[17];
rz(2.7119363974615514*pi) q[18];
rz(1.8901336558422006*pi) q[19];
rz(0.32044029018570086*pi) q[20];
rz(2.9896188386799007*pi) q[21];
rz(1.3138220516200931*pi) q[22];
rz(2.4726049340625167*pi) q[23];
rz(0.3659993848465266*pi) q[24];
rz(2.5515898278440083*pi) q[25];
rz(1.0142360913599475*pi) q[26];
rz(0.6216997991932995*pi) q[27];
rz(1.4147698850685*pi) q[28];
rz(0.5790070258216993*pi) q[29];
rz(2.7827031536940003*pi) q[30];
rz(1.9775581996758813*pi) q[31];
rz(3.010071166550256*pi) q[32];
rz(0.6685797703328191*pi) q[33];
rz(2.7424768988664994*pi) q[34];
rz(0.6414433571810001*pi) q[35];
rz(2.8402748704145857*pi) q[36];
rz(2.232232349135799*pi) q[37];
rz(2.0537698351371994*pi) q[38];
rz(1.8997431715194004*pi) q[39];
U1q(0.35256566228067*pi,0.386159745291056*pi) q[0];
U1q(1.33661614590037*pi,0.355328590269459*pi) q[1];
U1q(0.570902965551281*pi,1.671626254636138*pi) q[2];
U1q(1.44449468724544*pi,0.6808230788698499*pi) q[3];
U1q(1.44711094535565*pi,1.790574931295337*pi) q[4];
U1q(1.95588392915507*pi,1.111353479180839*pi) q[5];
U1q(0.237066971389136*pi,0.972166574482467*pi) q[6];
U1q(0.108678865384613*pi,1.752877448638096*pi) q[7];
U1q(0.964064025726886*pi,1.589804118352718*pi) q[8];
U1q(0.73990152288906*pi,0.130611205764719*pi) q[9];
U1q(0.522383913313181*pi,0.752280851678525*pi) q[10];
U1q(3.057130385445162*pi,1.4406093441794678*pi) q[11];
U1q(0.472851296067245*pi,0.513608268574603*pi) q[12];
U1q(0.369488798234213*pi,0.327271001135865*pi) q[13];
U1q(0.757307443042404*pi,1.777690464609831*pi) q[14];
U1q(0.13475590568513*pi,0.504095260709923*pi) q[15];
U1q(1.60800361009309*pi,1.5231831429166571*pi) q[16];
U1q(0.708611846668356*pi,0.254351614389537*pi) q[17];
U1q(0.705128497379431*pi,0.805926569099933*pi) q[18];
U1q(0.68813235837659*pi,1.614695210485511*pi) q[19];
U1q(0.549240895383064*pi,0.607152006533003*pi) q[20];
U1q(0.319317335703153*pi,0.916563165799397*pi) q[21];
U1q(0.376958833375452*pi,0.689616040823273*pi) q[22];
U1q(0.273781948656211*pi,1.550624974293003*pi) q[23];
U1q(3.084144066524416*pi,1.3664697188870951*pi) q[24];
U1q(0.0550100483525465*pi,1.514810965200135*pi) q[25];
U1q(3.490662803631904*pi,0.27389236614343*pi) q[26];
U1q(3.532250409294444*pi,0.740425287416457*pi) q[27];
U1q(3.2198767572040428*pi,0.457409793884065*pi) q[28];
U1q(0.266737229272407*pi,1.139288443028144*pi) q[29];
U1q(0.23085730975365*pi,0.720499370018151*pi) q[30];
U1q(0.142521776633297*pi,1.1141112822095*pi) q[31];
U1q(0.679728820211309*pi,0.698226856988435*pi) q[32];
U1q(3.168110083785445*pi,0.77775136795987*pi) q[33];
U1q(1.26813415307293*pi,1.212519639987055*pi) q[34];
U1q(0.397137192194134*pi,1.877728135672168*pi) q[35];
U1q(1.41008378392287*pi,0.53602767033472*pi) q[36];
U1q(1.1090892510351*pi,0.299207486651564*pi) q[37];
U1q(1.72646121688422*pi,0.970649509671577*pi) q[38];
U1q(0.500923799042404*pi,1.3759548005723379*pi) q[39];
RZZ(0.5*pi) q[0],q[4];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[28],q[3];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[6],q[17];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[22],q[9];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[12],q[27];
RZZ(0.5*pi) q[14],q[30];
RZZ(0.5*pi) q[15],q[33];
RZZ(0.5*pi) q[18],q[24];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[31],q[21];
RZZ(0.5*pi) q[34],q[23];
RZZ(0.5*pi) q[35],q[25];
RZZ(0.5*pi) q[39],q[38];
U1q(0.342298132846749*pi,0.44042957102246993*pi) q[0];
U1q(3.402498898037674*pi,0.5122349240222404*pi) q[1];
U1q(1.37415458426619*pi,1.7459461813585802*pi) q[2];
U1q(1.35677571670257*pi,1.6210937350669474*pi) q[3];
U1q(3.372192372234564*pi,0.14690974681731178*pi) q[4];
U1q(1.11133146879283*pi,0.9129048979270058*pi) q[5];
U1q(1.35973321104256*pi,1.506408603703756*pi) q[6];
U1q(0.496955286420695*pi,0.06430217740100996*pi) q[7];
U1q(0.491673242414619*pi,0.4461713086088701*pi) q[8];
U1q(1.55289718704682*pi,1.94839416284037*pi) q[9];
U1q(1.76163832260737*pi,1.31494269123217*pi) q[10];
U1q(1.30016295668443*pi,0.07538795061432513*pi) q[11];
U1q(0.334236898618689*pi,0.90895621590917*pi) q[12];
U1q(1.56613325806937*pi,1.594734784130675*pi) q[13];
U1q(0.664171476336456*pi,0.74185263029605*pi) q[14];
U1q(0.731927493298356*pi,1.2863996503544128*pi) q[15];
U1q(3.352890505092654*pi,0.35921051679613214*pi) q[16];
U1q(1.60383699595493*pi,1.863024739506651*pi) q[17];
U1q(0.758919197180287*pi,1.3589793465671152*pi) q[18];
U1q(0.253583434351411*pi,1.048768052637214*pi) q[19];
U1q(1.39467401499234*pi,0.6496042831053499*pi) q[20];
U1q(1.28841754218632*pi,0.83549520606596*pi) q[21];
U1q(1.42560468500957*pi,1.218469322107774*pi) q[22];
U1q(0.558602058636447*pi,1.78008326796304*pi) q[23];
U1q(3.419336155749411*pi,1.902754008124937*pi) q[24];
U1q(1.15655146592311*pi,0.60840474459667*pi) q[25];
U1q(3.172195592778927*pi,1.7558710742818273*pi) q[26];
U1q(1.42989256948574*pi,1.9186599793592114*pi) q[27];
U1q(1.43994401993188*pi,1.8686409413296996*pi) q[28];
U1q(3.27351363093205*pi,1.7444198081277502*pi) q[29];
U1q(0.538767057186834*pi,1.37895391987568*pi) q[30];
U1q(1.43019501939002*pi,1.9356891591774603*pi) q[31];
U1q(1.68112950892386*pi,1.35610115854898*pi) q[32];
U1q(1.80632290041267*pi,1.3015784688851855*pi) q[33];
U1q(3.384023258532713*pi,0.3203644065053879*pi) q[34];
U1q(1.77554230596264*pi,0.3694368356580999*pi) q[35];
U1q(1.52414822169632*pi,1.2008612752655186*pi) q[36];
U1q(1.83103018381399*pi,1.6916412310798699*pi) q[37];
U1q(3.545796935061798*pi,0.30406273743275236*pi) q[38];
U1q(1.73656267789117*pi,1.58162794506831*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[2],q[5];
RZZ(0.5*pi) q[19],q[3];
RZZ(0.5*pi) q[4],q[21];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[7],q[33];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[20];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[32],q[17];
RZZ(0.5*pi) q[22],q[26];
RZZ(0.5*pi) q[35],q[23];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[31],q[27];
U1q(3.305941738869808*pi,1.88236203273323*pi) q[0];
U1q(0.462604510419231*pi,0.4721180318600413*pi) q[1];
U1q(1.700418126428*pi,1.4008924863576961*pi) q[2];
U1q(1.38961204881495*pi,1.8971481396783076*pi) q[3];
U1q(0.554759427767341*pi,0.9901371475966028*pi) q[4];
U1q(0.285136898699028*pi,0.9618208171195959*pi) q[5];
U1q(1.88129531138775*pi,1.6742643935243458*pi) q[6];
U1q(0.426698339056876*pi,1.7355709806032698*pi) q[7];
U1q(1.65932803616109*pi,1.17321872748472*pi) q[8];
U1q(1.141246994994*pi,0.11272636009368298*pi) q[9];
U1q(1.79536663346314*pi,1.6098499205066084*pi) q[10];
U1q(3.629968827376369*pi,1.560509435483835*pi) q[11];
U1q(1.45664189273592*pi,0.55147104168745*pi) q[12];
U1q(3.584087727446161*pi,0.5105994024764402*pi) q[13];
U1q(1.75422768140977*pi,0.2827222999613299*pi) q[14];
U1q(1.83520306829262*pi,0.70882688672974*pi) q[15];
U1q(1.23576266177595*pi,0.935110916361495*pi) q[16];
U1q(3.529646596758596*pi,0.8388022350373461*pi) q[17];
U1q(1.75099552470567*pi,1.9216329271593777*pi) q[18];
U1q(1.53752951068865*pi,0.92395422291488*pi) q[19];
U1q(3.841907905194065*pi,0.5860107295961487*pi) q[20];
U1q(1.70679437643537*pi,1.819660110419485*pi) q[21];
U1q(1.28706344539486*pi,0.1051666084512326*pi) q[22];
U1q(1.35012504801009*pi,1.95840519628036*pi) q[23];
U1q(3.50157876698108*pi,1.2281518496171238*pi) q[24];
U1q(1.42923243945124*pi,1.6468008436308903*pi) q[25];
U1q(3.2802998322271*pi,0.438131487611078*pi) q[26];
U1q(0.263903367699081*pi,1.9019723555379144*pi) q[27];
U1q(0.837909537325797*pi,0.0485884654076747*pi) q[28];
U1q(1.38294309755894*pi,1.6085224513477208*pi) q[29];
U1q(0.423671394635641*pi,1.67646626627543*pi) q[30];
U1q(3.574962452492921*pi,1.938545934751814*pi) q[31];
U1q(3.362445133975592*pi,1.4519907635099538*pi) q[32];
U1q(0.576295906320182*pi,0.08547409311485937*pi) q[33];
U1q(1.46582652922303*pi,1.5370970017022216*pi) q[34];
U1q(3.5012436723048888*pi,1.2460849759308634*pi) q[35];
U1q(1.48093434268239*pi,1.7985096970083791*pi) q[36];
U1q(0.466194365792323*pi,0.7826815612203157*pi) q[37];
U1q(3.411183906268494*pi,0.9875100455521226*pi) q[38];
U1q(3.354559860539868*pi,1.3479740746223667*pi) q[39];
RZZ(0.5*pi) q[0],q[28];
RZZ(0.5*pi) q[1],q[30];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[38],q[3];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[24],q[5];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[10],q[31];
RZZ(0.5*pi) q[27],q[11];
RZZ(0.5*pi) q[34],q[13];
RZZ(0.5*pi) q[14],q[15];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[36],q[17];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[22],q[32];
RZZ(0.5*pi) q[23],q[25];
U1q(1.73321966308204*pi,0.9200170928649656*pi) q[0];
U1q(0.27716725022013*pi,0.9241880223370713*pi) q[1];
U1q(0.68434042538812*pi,1.8283482148340866*pi) q[2];
U1q(3.634888275434684*pi,0.004880693172289119*pi) q[3];
U1q(0.569663551730929*pi,0.9121624645876287*pi) q[4];
U1q(0.20427579258879*pi,1.5631930030354555*pi) q[5];
U1q(3.322194021290991*pi,0.2999987488618796*pi) q[6];
U1q(1.75666423333009*pi,1.9153224331854997*pi) q[7];
U1q(1.36872397142155*pi,1.7018509528646533*pi) q[8];
U1q(1.35751807611725*pi,0.052324440616703605*pi) q[9];
U1q(0.54952981561029*pi,1.4568096438467482*pi) q[10];
U1q(1.36853521214358*pi,0.8234922698441052*pi) q[11];
U1q(1.3307319014807*pi,1.1152310964634935*pi) q[12];
U1q(3.677055490336615*pi,0.4452621043557803*pi) q[13];
U1q(3.282286813131323*pi,0.37670740378524714*pi) q[14];
U1q(3.629189228321815*pi,1.3922092109515738*pi) q[15];
U1q(3.394301521540455*pi,0.21199395011235556*pi) q[16];
U1q(3.722790188261914*pi,0.43649932050202533*pi) q[17];
U1q(3.527250328412097*pi,1.8811254214267188*pi) q[18];
U1q(1.93484306499458*pi,1.9619633813608397*pi) q[19];
U1q(3.147879197126338*pi,1.9772601865472188*pi) q[20];
U1q(0.576078998600707*pi,0.886151080034133*pi) q[21];
U1q(1.66429143482149*pi,0.7249943627908126*pi) q[22];
U1q(1.49245226025979*pi,1.080096167112647*pi) q[23];
U1q(3.679541877154663*pi,1.5549540458802138*pi) q[24];
U1q(1.44937904479663*pi,1.9737418219934604*pi) q[25];
U1q(3.43191626405559*pi,1.857936544914442*pi) q[26];
U1q(0.196571053005968*pi,0.5436500647452375*pi) q[27];
U1q(1.47256452062374*pi,1.2502913009409955*pi) q[28];
U1q(1.73110820403992*pi,0.9894281462993408*pi) q[29];
U1q(1.44561395747073*pi,1.2464875612026098*pi) q[30];
U1q(3.59908469338748*pi,1.472738161866677*pi) q[31];
U1q(1.65908023144775*pi,0.8744793417572099*pi) q[32];
U1q(1.5274012190715*pi,1.9758664378359905*pi) q[33];
U1q(0.530623271815912*pi,1.1419439255548811*pi) q[34];
U1q(3.7063464701806392*pi,1.5225555515510836*pi) q[35];
U1q(3.876835355790051*pi,0.8465214936196945*pi) q[36];
U1q(0.192351609419782*pi,1.9761179828463957*pi) q[37];
U1q(1.61222758334347*pi,0.8261265987008048*pi) q[38];
U1q(1.63763000370626*pi,1.876743422655737*pi) q[39];
RZZ(0.5*pi) q[23],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[3],q[9];
RZZ(0.5*pi) q[27],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[28],q[8];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[26],q[13];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[16],q[30];
RZZ(0.5*pi) q[38],q[17];
RZZ(0.5*pi) q[19],q[33];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[29],q[39];
RZZ(0.5*pi) q[36],q[32];
U1q(0.62372037353425*pi,1.7264145988957758*pi) q[0];
U1q(1.56336705033495*pi,0.21004080590958107*pi) q[1];
U1q(1.89204032169585*pi,1.8966488630333167*pi) q[2];
U1q(3.888596002181833*pi,1.1551168131304594*pi) q[3];
U1q(0.208667723948972*pi,1.9616845128113347*pi) q[4];
U1q(1.32668484219216*pi,0.13851447305923603*pi) q[5];
U1q(1.83391109264378*pi,1.7210208147837225*pi) q[6];
U1q(3.355340155269384*pi,0.567002070067895*pi) q[7];
U1q(0.451361473035717*pi,0.22456374617445363*pi) q[8];
U1q(3.841805639531532*pi,1.1138808241635836*pi) q[9];
U1q(0.497602619324248*pi,0.8600298556320585*pi) q[10];
U1q(0.322999284190426*pi,0.9340350252347953*pi) q[11];
U1q(0.724150759437483*pi,1.5979789099232633*pi) q[12];
U1q(1.14513085950573*pi,0.8425367608444194*pi) q[13];
U1q(3.510200910227783*pi,0.8491954133740671*pi) q[14];
U1q(3.582273212380528*pi,1.1128605275155037*pi) q[15];
U1q(3.338035794128966*pi,1.7374875596348378*pi) q[16];
U1q(1.50425138068513*pi,1.2713063591338134*pi) q[17];
U1q(3.716118858096929*pi,0.6965458202518189*pi) q[18];
U1q(1.8312596775403*pi,1.7836039255573897*pi) q[19];
U1q(1.63348254337045*pi,1.1698129194602016*pi) q[20];
U1q(1.33552763927162*pi,1.377399632580615*pi) q[21];
U1q(3.132741316681248*pi,1.3531575493640053*pi) q[22];
U1q(1.75843178138169*pi,1.281692528024727*pi) q[23];
U1q(3.564788195339657*pi,1.8121340592599138*pi) q[24];
U1q(1.36220604554654*pi,1.896819609405556*pi) q[25];
U1q(1.58504085495418*pi,0.23800554458310108*pi) q[26];
U1q(0.23172888278585*pi,0.6016729661844744*pi) q[27];
U1q(3.658775207280813*pi,0.4406686031142435*pi) q[28];
U1q(3.094833718695735*pi,1.270173142564289*pi) q[29];
U1q(1.66673253281024*pi,1.0280419881692722*pi) q[30];
U1q(1.58519806383132*pi,1.6865509742615772*pi) q[31];
U1q(0.782032554871343*pi,1.500714700583647*pi) q[32];
U1q(1.24704256160929*pi,0.6894391237707955*pi) q[33];
U1q(0.338901393947434*pi,1.2293913942934713*pi) q[34];
U1q(3.459793839986065*pi,1.820753044777283*pi) q[35];
U1q(1.46986667118124*pi,1.2599217957634483*pi) q[36];
U1q(0.3270187592466*pi,0.41237377914263584*pi) q[37];
U1q(1.90093342982522*pi,0.48840089371944195*pi) q[38];
U1q(3.241175368142835*pi,0.27024376401732253*pi) q[39];
RZZ(0.5*pi) q[0],q[33];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[2],q[30];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[12],q[4];
RZZ(0.5*pi) q[39],q[6];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[38],q[9];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[11],q[21];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[35],q[14];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[25],q[17];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[34],q[22];
RZZ(0.5*pi) q[26],q[24];
RZZ(0.5*pi) q[29],q[37];
RZZ(0.5*pi) q[31],q[36];
U1q(0.351331760384051*pi,0.35192436316277576*pi) q[0];
U1q(1.42521837269279*pi,1.154413810687534*pi) q[1];
U1q(1.8720940801897*pi,1.470330979367394*pi) q[2];
U1q(1.31734672929832*pi,1.0605151890558062*pi) q[3];
U1q(0.540003242019346*pi,1.152704400545515*pi) q[4];
U1q(1.63375253799421*pi,0.2982056640451418*pi) q[5];
U1q(0.315974011254824*pi,1.3005227579402323*pi) q[6];
U1q(0.548526386905289*pi,1.6457134397525444*pi) q[7];
U1q(0.504558625111548*pi,1.938269728986894*pi) q[8];
U1q(3.1302879642710533*pi,0.5652521382290887*pi) q[9];
U1q(0.680033860790906*pi,1.1941313395079387*pi) q[10];
U1q(0.449348657341837*pi,0.4280340065671253*pi) q[11];
U1q(0.768270577978058*pi,1.5276690783801232*pi) q[12];
U1q(0.487512407501099*pi,1.5153522847887004*pi) q[13];
U1q(1.6650773023742*pi,0.6546247015557098*pi) q[14];
U1q(3.55040385136427*pi,1.8296521296037742*pi) q[15];
U1q(0.901165612404146*pi,1.1962838979011874*pi) q[16];
U1q(0.679125729430465*pi,1.8248864937950335*pi) q[17];
U1q(1.52329536138957*pi,1.6490859291604352*pi) q[18];
U1q(3.608061611433354*pi,1.2673831250415066*pi) q[19];
U1q(0.326027764877972*pi,0.34428431867821185*pi) q[20];
U1q(1.42840087094606*pi,0.17733980346268075*pi) q[21];
U1q(1.17914799199865*pi,0.7252635612317375*pi) q[22];
U1q(3.180133133484885*pi,0.9462530815740928*pi) q[23];
U1q(1.66619379879847*pi,0.049843803140434684*pi) q[24];
U1q(0.610821434144951*pi,1.7437392391138662*pi) q[25];
U1q(1.39122594361999*pi,1.4814738295019723*pi) q[26];
U1q(0.652127259386781*pi,1.9638387815184943*pi) q[27];
U1q(0.847685595591805*pi,0.46742431857532374*pi) q[28];
U1q(0.19338455718697*pi,0.14005339560456864*pi) q[29];
U1q(0.339648919573013*pi,0.712243793006432*pi) q[30];
U1q(0.910610596035432*pi,1.2197031972565302*pi) q[31];
U1q(0.666049130536585*pi,0.3790405694891166*pi) q[32];
U1q(0.644344578545044*pi,1.227989535683093*pi) q[33];
U1q(0.395264312440536*pi,1.3193735043617512*pi) q[34];
U1q(1.37329739744893*pi,1.473743137926962*pi) q[35];
U1q(0.715596723870153*pi,1.4234539484184783*pi) q[36];
U1q(0.404232654412166*pi,1.5961467952014363*pi) q[37];
U1q(0.509103315640634*pi,1.163713769846212*pi) q[38];
U1q(3.501892039774633*pi,0.40865270152824484*pi) q[39];
rz(3.6480756368372242*pi) q[0];
rz(2.845586189312466*pi) q[1];
rz(2.529669020632606*pi) q[2];
rz(0.9394848109441938*pi) q[3];
rz(2.847295599454485*pi) q[4];
rz(3.701794335954858*pi) q[5];
rz(0.6994772420597677*pi) q[6];
rz(2.3542865602474556*pi) q[7];
rz(0.06173027101310602*pi) q[8];
rz(1.4347478617709113*pi) q[9];
rz(0.8058686604920613*pi) q[10];
rz(1.5719659934328747*pi) q[11];
rz(0.4723309216198768*pi) q[12];
rz(0.4846477152112996*pi) q[13];
rz(1.3453752984442902*pi) q[14];
rz(0.1703478703962258*pi) q[15];
rz(2.8037161020988126*pi) q[16];
rz(2.1751135062049665*pi) q[17];
rz(0.35091407083956483*pi) q[18];
rz(0.7326168749584934*pi) q[19];
rz(3.655715681321788*pi) q[20];
rz(1.8226601965373193*pi) q[21];
rz(3.2747364387682625*pi) q[22];
rz(3.053746918425907*pi) q[23];
rz(3.9501561968595653*pi) q[24];
rz(2.256260760886134*pi) q[25];
rz(2.5185261704980277*pi) q[26];
rz(0.036161218481505664*pi) q[27];
rz(3.5325756814246763*pi) q[28];
rz(3.8599466043954314*pi) q[29];
rz(1.287756206993568*pi) q[30];
rz(2.78029680274347*pi) q[31];
rz(3.6209594305108834*pi) q[32];
rz(2.772010464316907*pi) q[33];
rz(2.680626495638249*pi) q[34];
rz(0.5262568620730379*pi) q[35];
rz(2.5765460515815217*pi) q[36];
rz(0.4038532047985637*pi) q[37];
rz(2.836286230153788*pi) q[38];
rz(3.591347298471755*pi) q[39];
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