OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
rz(1.0098106614032063*pi) q[0];
rz(1.4759515359965585*pi) q[1];
rz(0.482367729543623*pi) q[2];
rz(1.420314815209105*pi) q[3];
rz(2.6198343554632206*pi) q[4];
rz(0.291733048904396*pi) q[5];
rz(0.0276252608026417*pi) q[6];
rz(1.04202365062077*pi) q[7];
rz(3.193426708958498*pi) q[8];
rz(0.839703850138857*pi) q[9];
rz(1.10662984858575*pi) q[10];
rz(0.8038014939915776*pi) q[11];
rz(1.29498219810536*pi) q[12];
rz(3.347881980361686*pi) q[13];
rz(1.40755856421904*pi) q[14];
rz(1.6714120836563544*pi) q[15];
rz(3.524604597079636*pi) q[16];
rz(1.23855860954908*pi) q[17];
rz(1.42959740774627*pi) q[18];
rz(3.630273077004974*pi) q[19];
rz(1.1037654088360536*pi) q[20];
rz(0.835599881310501*pi) q[21];
rz(0.6647661447430255*pi) q[22];
rz(1.407128900902*pi) q[23];
rz(0.879738816021991*pi) q[24];
rz(0.6243893623077768*pi) q[25];
rz(0.66934780717749*pi) q[26];
rz(1.27379910409232*pi) q[27];
rz(3.8516852831792363*pi) q[28];
rz(3.6658716746850213*pi) q[29];
rz(1.21912780246354*pi) q[30];
rz(3.819132825068665*pi) q[31];
rz(2.6027324651863744*pi) q[32];
rz(0.04536897623526792*pi) q[33];
rz(3.499947074414144*pi) q[34];
rz(3.892987816560367*pi) q[35];
rz(3.809052731301854*pi) q[36];
rz(0.9385154333173891*pi) q[37];
rz(3.615304604387319*pi) q[38];
rz(1.33413898646739*pi) q[39];
U1q(1.39214404929016*pi,0.964473746534308*pi) q[0];
U1q(1.25822943103623*pi,1.34371327511622*pi) q[1];
U1q(0.851566192540794*pi,0.0565152105843649*pi) q[2];
U1q(1.84800284554759*pi,1.27391719946458*pi) q[3];
U1q(1.49242999345706*pi,1.45370666973866*pi) q[4];
U1q(0.132104637245359*pi,1.7917569132502509*pi) q[5];
U1q(0.408031276888322*pi,1.16626049338457*pi) q[6];
U1q(0.351717044981423*pi,1.22321327007265*pi) q[7];
U1q(0.721111657692621*pi,0.11876936603474*pi) q[8];
U1q(0.131306494990944*pi,0.455012807134547*pi) q[9];
U1q(0.616653921636664*pi,0.382885173308276*pi) q[10];
U1q(3.229262492704737*pi,1.819807461225568*pi) q[11];
U1q(0.616198681464391*pi,1.37945203175758*pi) q[12];
U1q(0.906333978690642*pi,0.198709878645609*pi) q[13];
U1q(0.187283358780977*pi,1.54385164216481*pi) q[14];
U1q(1.16653761786961*pi,1.07204190815154*pi) q[15];
U1q(1.93839894939992*pi,0.272131463893624*pi) q[16];
U1q(0.383120073654909*pi,1.04845217980123*pi) q[17];
U1q(0.607612729371205*pi,1.0357525859276*pi) q[18];
U1q(0.665657624620561*pi,1.731170104596073*pi) q[19];
U1q(1.42794782830314*pi,1.27502133678512*pi) q[20];
U1q(0.865332922200447*pi,1.08176707427174*pi) q[21];
U1q(1.68150111876545*pi,1.9553217817329163*pi) q[22];
U1q(0.373625307949664*pi,1.34899518965904*pi) q[23];
U1q(0.280230097925593*pi,0.546769445010952*pi) q[24];
U1q(3.404702928764677*pi,1.801272242446085*pi) q[25];
U1q(0.419494627771888*pi,0.165441127228729*pi) q[26];
U1q(0.028545993005726*pi,0.0138684527830613*pi) q[27];
U1q(1.36086058624348*pi,0.618156060656227*pi) q[28];
U1q(1.69676365488588*pi,0.100398489748148*pi) q[29];
U1q(0.444515943518767*pi,1.52140024309536*pi) q[30];
U1q(0.343770247489106*pi,0.729710512418353*pi) q[31];
U1q(3.903853844510942*pi,1.401178238558471*pi) q[32];
U1q(1.86317216223535*pi,1.087916304812172*pi) q[33];
U1q(0.500358870514724*pi,1.867169345474363*pi) q[34];
U1q(0.737508964603436*pi,0.538294986655644*pi) q[35];
U1q(1.58583457136554*pi,1.2400661965629611*pi) q[36];
U1q(1.47001391934915*pi,1.23111472333864*pi) q[37];
U1q(0.618873470926849*pi,1.897335558844594*pi) q[38];
U1q(0.51399658147533*pi,1.9325127833082976*pi) q[39];
RZZ(0.0*pi) q[0],q[5];
RZZ(0.0*pi) q[3],q[1];
RZZ(0.0*pi) q[2],q[29];
RZZ(0.0*pi) q[4],q[30];
RZZ(0.0*pi) q[32],q[6];
RZZ(0.0*pi) q[11],q[7];
RZZ(0.0*pi) q[8],q[15];
RZZ(0.0*pi) q[23],q[9];
RZZ(0.0*pi) q[25],q[10];
RZZ(0.0*pi) q[12],q[19];
RZZ(0.0*pi) q[20],q[13];
RZZ(0.0*pi) q[14],q[31];
RZZ(0.0*pi) q[16],q[33];
RZZ(0.0*pi) q[17],q[28];
RZZ(0.0*pi) q[18],q[37];
RZZ(0.0*pi) q[21],q[38];
RZZ(0.0*pi) q[22],q[35];
RZZ(0.0*pi) q[34],q[24];
RZZ(0.0*pi) q[26],q[39];
RZZ(0.0*pi) q[27],q[36];
rz(0.138628966664803*pi) q[0];
rz(0.772535011805498*pi) q[1];
rz(3.9545381973424014*pi) q[2];
rz(0.914535279066162*pi) q[3];
rz(1.16046747435206*pi) q[4];
rz(0.101186994176903*pi) q[5];
rz(3.880090461859587*pi) q[6];
rz(1.06435115762164*pi) q[7];
rz(2.78727827763959*pi) q[8];
rz(0.31645779754921*pi) q[9];
rz(3.832119936632607*pi) q[10];
rz(0.432193205646881*pi) q[11];
rz(3.874033180010865*pi) q[12];
rz(0.0709186396650535*pi) q[13];
rz(1.33124634879098*pi) q[14];
rz(2.41429201723875*pi) q[15];
rz(0.429986051346343*pi) q[16];
rz(3.7458410540790332*pi) q[17];
rz(3.701825971535562*pi) q[18];
rz(1.27297436331925*pi) q[19];
rz(0.828387479968874*pi) q[20];
rz(0.0901448251692032*pi) q[21];
rz(0.734865319024364*pi) q[22];
rz(0.687404340885491*pi) q[23];
rz(0.407750166879428*pi) q[24];
rz(0.529233605378499*pi) q[25];
rz(3.759036409480703*pi) q[26];
rz(0.701992251353017*pi) q[27];
rz(3.9988667186532103*pi) q[28];
rz(2.44098939247304*pi) q[29];
rz(3.9253259613355933*pi) q[30];
rz(0.271995566115886*pi) q[31];
rz(3.893830433879597*pi) q[32];
rz(1.04994208156928*pi) q[33];
rz(0.0450202250003742*pi) q[34];
rz(3.9944090328654958*pi) q[35];
rz(0.693222433540929*pi) q[36];
rz(3.9936193787237184*pi) q[37];
rz(0.553817730952688*pi) q[38];
rz(0.240516586233636*pi) q[39];
U1q(0.691983405421809*pi,0.140901492766329*pi) q[0];
U1q(0.482618890725651*pi,1.6063457485427999*pi) q[1];
U1q(0.166918864306702*pi,0.66528174741214*pi) q[2];
U1q(0.346091746056296*pi,0.251875023326896*pi) q[3];
U1q(0.69866167087554*pi,0.961497343559493*pi) q[4];
U1q(0.400077604865545*pi,0.433034081546318*pi) q[5];
U1q(0.431380642216402*pi,1.538133907894724*pi) q[6];
U1q(0.521737757404218*pi,1.04840736969079*pi) q[7];
U1q(0.633610372972421*pi,1.389957134927287*pi) q[8];
U1q(0.297363494982272*pi,1.919797499993368*pi) q[9];
U1q(0.603158327625636*pi,0.110217473924476*pi) q[10];
U1q(0.830235584605934*pi,0.779593846459187*pi) q[11];
U1q(0.403855375292516*pi,0.308773173466939*pi) q[12];
U1q(0.245959101444573*pi,0.53007311756871*pi) q[13];
U1q(0.430120042558485*pi,1.06562892721365*pi) q[14];
U1q(0.616910526582536*pi,1.166658355405919*pi) q[15];
U1q(0.19651941867471*pi,1.1936622917842*pi) q[16];
U1q(0.880431088892944*pi,1.872584257839195*pi) q[17];
U1q(0.790056366005632*pi,1.787524766039131*pi) q[18];
U1q(0.676039153488536*pi,1.29405098475872*pi) q[19];
U1q(0.500604113061285*pi,1.2942753158502*pi) q[20];
U1q(0.0795138341982985*pi,1.629856465136419*pi) q[21];
U1q(0.65564596957206*pi,1.00250795671319*pi) q[22];
U1q(0.278876597673316*pi,1.59880769747374*pi) q[23];
U1q(0.506772105676516*pi,0.717865476224118*pi) q[24];
U1q(0.787123543181131*pi,0.714633282371764*pi) q[25];
U1q(0.233397166154377*pi,1.3405093211876449*pi) q[26];
U1q(0.403436398112674*pi,1.617222688306644*pi) q[27];
U1q(0.154135902898667*pi,1.598823324812765*pi) q[28];
U1q(0.851303751768333*pi,0.99763258972398*pi) q[29];
U1q(0.141040540386438*pi,1.848574236908843*pi) q[30];
U1q(0.403827971610169*pi,0.667425688893325*pi) q[31];
U1q(0.315720751301075*pi,1.471989232367894*pi) q[32];
U1q(0.472189796828889*pi,0.311556197024942*pi) q[33];
U1q(0.62371345887518*pi,0.743033005585758*pi) q[34];
U1q(0.716579178723785*pi,1.886588445979912*pi) q[35];
U1q(0.795482028248776*pi,0.366033327648171*pi) q[36];
U1q(0.34172271052756*pi,0.976954310615212*pi) q[37];
U1q(0.381500628561355*pi,1.991519009964721*pi) q[38];
U1q(0.264378963047477*pi,0.711543126320225*pi) q[39];
RZZ(0.0*pi) q[0],q[21];
RZZ(0.0*pi) q[14],q[1];
RZZ(0.0*pi) q[2],q[11];
RZZ(0.0*pi) q[3],q[15];
RZZ(0.0*pi) q[4],q[28];
RZZ(0.0*pi) q[5],q[18];
RZZ(0.0*pi) q[26],q[6];
RZZ(0.0*pi) q[25],q[7];
RZZ(0.0*pi) q[8],q[38];
RZZ(0.0*pi) q[13],q[9];
RZZ(0.0*pi) q[31],q[10];
RZZ(0.0*pi) q[12],q[29];
RZZ(0.0*pi) q[16],q[36];
RZZ(0.0*pi) q[23],q[17];
RZZ(0.0*pi) q[33],q[19];
RZZ(0.0*pi) q[32],q[20];
RZZ(0.0*pi) q[22],q[37];
RZZ(0.0*pi) q[30],q[24];
RZZ(0.0*pi) q[39],q[27];
RZZ(0.0*pi) q[35],q[34];
rz(3.727879540931064*pi) q[0];
rz(1.6389451940596*pi) q[1];
rz(0.932076282158494*pi) q[2];
rz(1.87457806915692*pi) q[3];
rz(3.842039847474835*pi) q[4];
rz(0.222150195323871*pi) q[5];
rz(0.245299626249115*pi) q[6];
rz(0.307563480291172*pi) q[7];
rz(0.430300234960343*pi) q[8];
rz(0.611524657762603*pi) q[9];
rz(1.89725226854179*pi) q[10];
rz(0.300021328132887*pi) q[11];
rz(3.494332966835451*pi) q[12];
rz(0.309532766882813*pi) q[13];
rz(1.68096366448914*pi) q[14];
rz(3.402582270071237*pi) q[15];
rz(0.821934356830436*pi) q[16];
rz(0.687008912363604*pi) q[17];
rz(1.7487291602577*pi) q[18];
rz(2.5635172139751*pi) q[19];
rz(0.78516656453198*pi) q[20];
rz(0.496530652478654*pi) q[21];
rz(0.0276108810041799*pi) q[22];
rz(3.9263620116430635*pi) q[23];
rz(0.347723340390895*pi) q[24];
rz(3.688545115205916*pi) q[25];
rz(3.33648579891601*pi) q[26];
rz(1.20116684975788*pi) q[27];
rz(3.798468093260128*pi) q[28];
rz(1.37468980445759*pi) q[29];
rz(0.518783967380018*pi) q[30];
rz(3.62367679672574*pi) q[31];
rz(0.508232633039179*pi) q[32];
rz(3.542523791342888*pi) q[33];
rz(0.266374962064559*pi) q[34];
rz(0.16416514419332*pi) q[35];
rz(1.39444927908254*pi) q[36];
rz(0.0462546831244447*pi) q[37];
rz(0.59237328610573*pi) q[38];
rz(2.76385729670212*pi) q[39];
U1q(0.046240875441722*pi,1.630082784647227*pi) q[0];
U1q(0.5076835307782*pi,1.22513577627276*pi) q[1];
U1q(0.448090010676017*pi,1.244242351887*pi) q[2];
U1q(0.54580737715873*pi,0.803494746217048*pi) q[3];
U1q(0.248685613590269*pi,0.852299982681867*pi) q[4];
U1q(0.421921720881417*pi,0.927632815370594*pi) q[5];
U1q(0.67015298167834*pi,0.484133591940969*pi) q[6];
U1q(0.376112360060454*pi,0.716909326790821*pi) q[7];
U1q(0.316365171744106*pi,1.14813172376408*pi) q[8];
U1q(0.339437294681287*pi,1.01228929800036*pi) q[9];
U1q(0.849940510456092*pi,1.43734752328693*pi) q[10];
U1q(0.884404724972121*pi,0.698252357469744*pi) q[11];
U1q(0.764231261752682*pi,0.080293930373146*pi) q[12];
U1q(0.947825406132574*pi,0.522873007399899*pi) q[13];
U1q(0.617179027907785*pi,0.771697233849256*pi) q[14];
U1q(0.671238235224554*pi,0.447961618446378*pi) q[15];
U1q(0.55071466281242*pi,1.07146859783482*pi) q[16];
U1q(0.623162176675706*pi,0.706673951178615*pi) q[17];
U1q(0.676997094114455*pi,0.745340500273925*pi) q[18];
U1q(0.889022034767526*pi,1.377682334366353*pi) q[19];
U1q(0.239084534169391*pi,0.995072837242243*pi) q[20];
U1q(0.335177929003166*pi,1.752825601901079*pi) q[21];
U1q(0.86542833717611*pi,1.841438010984579*pi) q[22];
U1q(0.462113876602056*pi,0.41965091123202*pi) q[23];
U1q(0.0149586884789583*pi,1.614307389071942*pi) q[24];
U1q(0.401925613646013*pi,1.16046785776725*pi) q[25];
U1q(0.679312424461396*pi,1.9702651227745256*pi) q[26];
U1q(0.464686040267148*pi,1.21766583726867*pi) q[27];
U1q(0.678797652027462*pi,0.557357648964716*pi) q[28];
U1q(0.172506742347976*pi,1.9008034435567365*pi) q[29];
U1q(0.576081192458006*pi,0.212903884921364*pi) q[30];
U1q(0.670866440777589*pi,0.307353039489553*pi) q[31];
U1q(0.863462451011497*pi,0.604812832203151*pi) q[32];
U1q(0.258462893378792*pi,0.418539779036111*pi) q[33];
U1q(0.302727335705219*pi,1.12497742751857*pi) q[34];
U1q(0.420594464118319*pi,0.472612225341707*pi) q[35];
U1q(0.477840741579022*pi,1.9593482981573533*pi) q[36];
U1q(0.261977015939867*pi,1.785285863526771*pi) q[37];
U1q(0.759000771038006*pi,0.533215155937951*pi) q[38];
U1q(0.753387901095348*pi,1.809637331068536*pi) q[39];
RZZ(0.0*pi) q[0],q[15];
RZZ(0.0*pi) q[11],q[1];
RZZ(0.0*pi) q[39],q[2];
RZZ(0.0*pi) q[5],q[3];
RZZ(0.0*pi) q[32],q[4];
RZZ(0.0*pi) q[6],q[20];
RZZ(0.0*pi) q[24],q[7];
RZZ(0.0*pi) q[8],q[12];
RZZ(0.0*pi) q[37],q[9];
RZZ(0.0*pi) q[26],q[10];
RZZ(0.0*pi) q[14],q[13];
RZZ(0.0*pi) q[22],q[16];
RZZ(0.0*pi) q[31],q[17];
RZZ(0.0*pi) q[18],q[36];
RZZ(0.0*pi) q[38],q[19];
RZZ(0.0*pi) q[27],q[21];
RZZ(0.0*pi) q[25],q[23];
RZZ(0.0*pi) q[35],q[28];
RZZ(0.0*pi) q[30],q[29];
RZZ(0.0*pi) q[34],q[33];
rz(0.292963650873524*pi) q[0];
rz(3.795614782629378*pi) q[1];
rz(1.00687557931244*pi) q[2];
rz(0.516115752589119*pi) q[3];
rz(1.40959839467772*pi) q[4];
rz(0.038096191988375*pi) q[5];
rz(1.18385489390811*pi) q[6];
rz(1.42294965472712*pi) q[7];
rz(2.61085969792977*pi) q[8];
rz(0.625664859123035*pi) q[9];
rz(0.466422528365047*pi) q[10];
rz(1.95920379198391*pi) q[11];
rz(0.681171829191024*pi) q[12];
rz(0.691360831821902*pi) q[13];
rz(0.505207225944361*pi) q[14];
rz(3.7741228141138*pi) q[15];
rz(0.239352761420908*pi) q[16];
rz(1.26064967956799*pi) q[17];
rz(2.7013757336430197*pi) q[18];
rz(3.099417379971335*pi) q[19];
rz(0.0084192697646287*pi) q[20];
rz(0.156257459710595*pi) q[21];
rz(0.350858719675401*pi) q[22];
rz(0.195486767391467*pi) q[23];
rz(3.898681556827289*pi) q[24];
rz(0.0801171841697788*pi) q[25];
rz(0.612322914966221*pi) q[26];
rz(0.656506094834566*pi) q[27];
rz(2.55760161634994*pi) q[28];
rz(2.6869625280575997*pi) q[29];
rz(3.537183411382529*pi) q[30];
rz(1.70543170754778*pi) q[31];
rz(2.40840377522948*pi) q[32];
rz(2.21023677963046*pi) q[33];
rz(0.964799301096842*pi) q[34];
rz(3.9357190258870487*pi) q[35];
rz(2.9615286048004*pi) q[36];
rz(3.895527826900259*pi) q[37];
rz(1.20679188848113*pi) q[38];
rz(0.997844513537644*pi) q[39];
U1q(0.81401182765335*pi,0.359909788311038*pi) q[0];
U1q(0.6629898978063*pi,0.519737455808871*pi) q[1];
U1q(0.451626530949723*pi,0.865101390348446*pi) q[2];
U1q(0.302834787238024*pi,0.333203449568029*pi) q[3];
U1q(0.777712262532666*pi,1.30295302224929*pi) q[4];
U1q(0.742996763651129*pi,1.9574441165358203*pi) q[5];
U1q(0.954628609780928*pi,1.26154089732791*pi) q[6];
U1q(0.377091791360932*pi,1.31256889792479*pi) q[7];
U1q(0.49387309892529*pi,1.299974632789773*pi) q[8];
U1q(0.280875043871403*pi,0.125238514993227*pi) q[9];
U1q(0.276618557527448*pi,0.771388495129336*pi) q[10];
U1q(0.712776673750422*pi,1.39743870441589*pi) q[11];
U1q(0.162286135163275*pi,0.492368454977716*pi) q[12];
U1q(0.880156694638744*pi,0.82876414230422*pi) q[13];
U1q(0.549311199743248*pi,0.190992413710927*pi) q[14];
U1q(0.308970404252215*pi,1.1163125610648281*pi) q[15];
U1q(0.411922040555941*pi,1.500345328406255*pi) q[16];
U1q(0.449371968061286*pi,0.898845440499245*pi) q[17];
U1q(0.476707532519951*pi,1.9396145014622368*pi) q[18];
U1q(0.611169995362354*pi,0.15637023988488*pi) q[19];
U1q(0.239781401321305*pi,1.707764779750808*pi) q[20];
U1q(0.408450033590771*pi,1.04435812206999*pi) q[21];
U1q(0.13316746863021*pi,0.170313270621996*pi) q[22];
U1q(0.471047314634535*pi,0.828126652169145*pi) q[23];
U1q(0.174375652422409*pi,1.60676654816454*pi) q[24];
U1q(0.337546894663869*pi,1.84228497446102*pi) q[25];
U1q(0.748611531076106*pi,0.230539443812506*pi) q[26];
U1q(0.290738843172127*pi,0.182131663686165*pi) q[27];
U1q(0.523138451709398*pi,1.429942688177581*pi) q[28];
U1q(0.804600705233*pi,1.397184841959239*pi) q[29];
U1q(0.459007419183939*pi,0.273339557043936*pi) q[30];
U1q(0.615362858969238*pi,0.982523850106673*pi) q[31];
U1q(0.786823061822436*pi,1.876065495862931*pi) q[32];
U1q(0.884310596109039*pi,1.103857432055026*pi) q[33];
U1q(0.294069614746238*pi,1.9931415437550295*pi) q[34];
U1q(0.756360393095086*pi,0.153650554579236*pi) q[35];
U1q(0.958650875667967*pi,1.716653639571617*pi) q[36];
U1q(0.295728997709378*pi,0.118401585944272*pi) q[37];
U1q(0.430236658948185*pi,0.167455901729596*pi) q[38];
U1q(0.0731917359906802*pi,1.706821601058541*pi) q[39];
RZZ(0.0*pi) q[0],q[19];
RZZ(0.0*pi) q[30],q[1];
RZZ(0.0*pi) q[2],q[17];
RZZ(0.0*pi) q[3],q[9];
RZZ(0.0*pi) q[4],q[7];
RZZ(0.0*pi) q[39],q[5];
RZZ(0.0*pi) q[23],q[6];
RZZ(0.0*pi) q[8],q[18];
RZZ(0.0*pi) q[10],q[28];
RZZ(0.0*pi) q[31],q[11];
RZZ(0.0*pi) q[12],q[34];
RZZ(0.0*pi) q[29],q[13];
RZZ(0.0*pi) q[14],q[38];
RZZ(0.0*pi) q[15],q[24];
RZZ(0.0*pi) q[16],q[21];
RZZ(0.0*pi) q[20],q[37];
RZZ(0.0*pi) q[26],q[22];
RZZ(0.0*pi) q[25],q[32];
RZZ(0.0*pi) q[27],q[33];
RZZ(0.0*pi) q[35],q[36];
rz(2.2125131576557*pi) q[0];
rz(3.614281975116302*pi) q[1];
rz(0.388602095950445*pi) q[2];
rz(0.636477706368501*pi) q[3];
rz(1.29445125328302*pi) q[4];
rz(1.24564280958289*pi) q[5];
rz(2.17632401830189*pi) q[6];
rz(0.207517169560756*pi) q[7];
rz(0.976832463097531*pi) q[8];
rz(3.863377908536038*pi) q[9];
rz(1.77241426246599*pi) q[10];
rz(2.0421194917502197*pi) q[11];
rz(0.0135099956328006*pi) q[12];
rz(1.10766528343268*pi) q[13];
rz(1.43449169794759*pi) q[14];
rz(0.762572980063163*pi) q[15];
rz(0.205566593255552*pi) q[16];
rz(1.16788043845343*pi) q[17];
rz(2.79824765480094*pi) q[18];
rz(0.7818226902738199*pi) q[19];
rz(2.84898732396447*pi) q[20];
rz(0.088944846380309*pi) q[21];
rz(1.6945623807515098*pi) q[22];
rz(2.27088563236096*pi) q[23];
rz(2.54975641162486*pi) q[24];
rz(3.194918596532838*pi) q[25];
rz(0.326787779570195*pi) q[26];
rz(0.801396183783123*pi) q[27];
rz(3.455354633149394*pi) q[28];
rz(2.10614061127899*pi) q[29];
rz(3.729818451388341*pi) q[30];
rz(2.29726968843792*pi) q[31];
rz(2.5827066690577603*pi) q[32];
rz(0.57009616858948*pi) q[33];
rz(3.654057092980536*pi) q[34];
rz(0.255312542455036*pi) q[35];
rz(0.9662207511129601*pi) q[36];
rz(0.635341855721425*pi) q[37];
rz(0.911433191731753*pi) q[38];
rz(1.0982515584827*pi) q[39];
U1q(3.191599935186394*pi,1.87044052673889*pi) q[0];
U1q(3.9400294899062804*pi,1.06164247762801*pi) q[1];
U1q(3.642535095651372*pi,1.844022059591451*pi) q[2];
U1q(3.2421406750018322*pi,0.657243495133637*pi) q[3];
U1q(3.364770272212306*pi,1.28257075461425*pi) q[4];
U1q(3.635080784170938*pi,1.647451909946506*pi) q[5];
U1q(3.225928395622388*pi,1.7943249521398341*pi) q[6];
U1q(3.832663620712252*pi,0.67831661644855*pi) q[7];
U1q(3.4281247277098412*pi,1.02409621416731*pi) q[8];
U1q(3.669514883671702*pi,1.438182679178809*pi) q[9];
U1q(3.681303845679939*pi,0.849621928425032*pi) q[10];
U1q(3.2121863389523497*pi,0.0622075324128224*pi) q[11];
U1q(3.3823786504559727*pi,1.0556236289622*pi) q[12];
U1q(3.090061453245205*pi,1.22758670993576*pi) q[13];
U1q(3.394262777910719*pi,0.439759215126795*pi) q[14];
U1q(3.184632638608806*pi,1.636022068045711*pi) q[15];
U1q(3.678905617126741*pi,0.469467393758038*pi) q[16];
U1q(3.35149278661683*pi,0.956117794552837*pi) q[17];
U1q(3.426030300723533*pi,0.24184713067003*pi) q[18];
U1q(3.421585403608619*pi,0.39544077124394006*pi) q[19];
U1q(3.958058537063375*pi,1.663163529442127*pi) q[20];
U1q(3.161023307251715*pi,0.113321969991874*pi) q[21];
U1q(3.559481904356574*pi,0.44610997487566006*pi) q[22];
U1q(3.683421453256833*pi,0.621167505288936*pi) q[23];
U1q(3.2941325231524807*pi,1.5904572195227629*pi) q[24];
U1q(3.314082955531406*pi,1.170494333544597*pi) q[25];
U1q(3.1496899846784308*pi,1.268288178913515*pi) q[26];
U1q(3.584707517598058*pi,0.318218292324288*pi) q[27];
U1q(3.519785459495391*pi,0.0345973963132913*pi) q[28];
U1q(3.692129172145472*pi,1.444176094935*pi) q[29];
U1q(3.688405031330946*pi,0.92707365489241*pi) q[30];
U1q(3.353361710587269*pi,0.5533008191217901*pi) q[31];
U1q(3.151199109932116*pi,1.874449274340336*pi) q[32];
U1q(3.482564911085562*pi,0.539553564494947*pi) q[33];
U1q(3.343538888878829*pi,1.518598057935818*pi) q[34];
U1q(3.538543422941212*pi,1.04636788871987*pi) q[35];
U1q(3.224626654505086*pi,0.9205839855862501*pi) q[36];
U1q(3.619828303727169*pi,1.702497279825057*pi) q[37];
U1q(3.229034936497581*pi,1.101267145106958*pi) q[38];
U1q(3.775830568559431*pi,1.726362416542033*pi) q[39];
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
