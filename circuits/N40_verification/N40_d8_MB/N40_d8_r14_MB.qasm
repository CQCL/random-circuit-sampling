OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(1.62067688064346*pi,1.3516045635983187*pi) q[0];
U1q(1.87498564563381*pi,1.5097008768242766*pi) q[1];
U1q(1.65664470503468*pi,1.7600737762688745*pi) q[2];
U1q(0.801938737125873*pi,0.980301793012457*pi) q[3];
U1q(3.471563828240787*pi,1.5021461404175624*pi) q[4];
U1q(1.61609321619081*pi,1.1979828815702014*pi) q[5];
U1q(1.33038049750369*pi,1.4323955578231196*pi) q[6];
U1q(0.720674998301511*pi,0.595639810234213*pi) q[7];
U1q(0.215532984129995*pi,1.572400977224488*pi) q[8];
U1q(0.101100557068128*pi,0.693706858618629*pi) q[9];
U1q(0.259767938612748*pi,1.9342413191048282*pi) q[10];
U1q(1.38623502317946*pi,1.3599800338153163*pi) q[11];
U1q(0.126775135125322*pi,1.07621148696206*pi) q[12];
U1q(0.384974224745688*pi,0.271521076190913*pi) q[13];
U1q(1.48499737422857*pi,0.6071426727230322*pi) q[14];
U1q(1.30373460900054*pi,1.817019462223923*pi) q[15];
U1q(1.79054062003504*pi,0.0029528516721731676*pi) q[16];
U1q(3.7064758503465*pi,1.1214309991687466*pi) q[17];
U1q(3.440141357922241*pi,1.2314042007860853*pi) q[18];
U1q(0.565909883502756*pi,0.307433067505691*pi) q[19];
U1q(0.526885386125895*pi,0.8170642407562501*pi) q[20];
U1q(0.520765512026259*pi,0.556578551864178*pi) q[21];
U1q(1.2605663083892*pi,0.11807340691207847*pi) q[22];
U1q(1.73916870839652*pi,1.4709393008854819*pi) q[23];
U1q(1.24863530427561*pi,0.6420570667721476*pi) q[24];
U1q(1.77237673123335*pi,0.3858334579127239*pi) q[25];
U1q(0.648599981209124*pi,0.8496726521953899*pi) q[26];
U1q(1.75659472223703*pi,0.18381078231981518*pi) q[27];
U1q(0.148523813423687*pi,0.300925136328704*pi) q[28];
U1q(1.24871783219827*pi,0.4284491884245993*pi) q[29];
U1q(3.520987685097743*pi,0.8769180685011966*pi) q[30];
U1q(1.38980826038554*pi,0.16188152355307517*pi) q[31];
U1q(0.597824056092767*pi,0.195796644170186*pi) q[32];
U1q(0.722065998063701*pi,1.508665507465242*pi) q[33];
U1q(1.83731786989461*pi,1.5402140210084354*pi) q[34];
U1q(1.33305187356399*pi,1.3533576978612558*pi) q[35];
U1q(1.47335069412627*pi,0.04780401051444766*pi) q[36];
U1q(1.45384319890066*pi,0.4745614710116388*pi) q[37];
U1q(0.423587663986964*pi,1.4366255155394478*pi) q[38];
U1q(0.73520645555331*pi,0.384644673670882*pi) q[39];
RZZ(0.5*pi) q[0],q[38];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[2],q[25];
RZZ(0.5*pi) q[3],q[26];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[9],q[37];
RZZ(0.5*pi) q[22],q[10];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[24],q[39];
RZZ(0.5*pi) q[27],q[32];
RZZ(0.5*pi) q[28],q[34];
U1q(0.29614156020767*pi,1.8476626819375288*pi) q[0];
U1q(0.326215791658225*pi,1.7321318125027565*pi) q[1];
U1q(0.496097589224354*pi,0.9308276286237895*pi) q[2];
U1q(0.689946932281283*pi,0.9921199979558799*pi) q[3];
U1q(0.52918735031052*pi,0.8917977972929516*pi) q[4];
U1q(0.564658216498694*pi,1.5014953996939613*pi) q[5];
U1q(0.384718105651078*pi,0.6191553865155699*pi) q[6];
U1q(0.0688251877190571*pi,1.8511051853529201*pi) q[7];
U1q(0.649557899168237*pi,0.5610954838365201*pi) q[8];
U1q(0.56838998458142*pi,0.0698029267387501*pi) q[9];
U1q(0.447980605111878*pi,1.29202441090692*pi) q[10];
U1q(0.619804168214141*pi,1.668769495629876*pi) q[11];
U1q(0.378248985752873*pi,0.91571501009245*pi) q[12];
U1q(0.0847288239837027*pi,0.7987064965549302*pi) q[13];
U1q(0.438521943703149*pi,0.9406311741514222*pi) q[14];
U1q(0.389823480336947*pi,0.6504745879859928*pi) q[15];
U1q(0.1054683536353*pi,1.0858646539376162*pi) q[16];
U1q(0.474277901560595*pi,0.03635424439012658*pi) q[17];
U1q(0.648139454694005*pi,0.4511976939502884*pi) q[18];
U1q(0.845477349980248*pi,0.01072480527658004*pi) q[19];
U1q(0.539275467588588*pi,0.4686346314893499*pi) q[20];
U1q(0.623039035449038*pi,0.185779543860993*pi) q[21];
U1q(0.113981964962647*pi,0.6693344632143186*pi) q[22];
U1q(0.508001173943114*pi,1.7338041810295222*pi) q[23];
U1q(0.746002460993178*pi,0.6363266001041779*pi) q[24];
U1q(0.188178033242543*pi,0.9935153298695338*pi) q[25];
U1q(0.814376833011864*pi,1.29628386057201*pi) q[26];
U1q(0.536630855526816*pi,0.8147191001961849*pi) q[27];
U1q(0.608820104790505*pi,0.9569128001448399*pi) q[28];
U1q(0.501513729132912*pi,1.290241254656499*pi) q[29];
U1q(0.495821930062629*pi,1.0597397678538967*pi) q[30];
U1q(0.865043650987482*pi,0.8577429857626253*pi) q[31];
U1q(0.145522486024939*pi,0.8741482621591099*pi) q[32];
U1q(0.10071673522969*pi,1.9211065625366297*pi) q[33];
U1q(0.340734404933302*pi,1.9273392291935156*pi) q[34];
U1q(0.319200363660073*pi,1.1400070111094758*pi) q[35];
U1q(0.252994102526987*pi,1.5222202456736778*pi) q[36];
U1q(0.289986196257993*pi,0.16854224046461863*pi) q[37];
U1q(0.283661430432589*pi,1.8820126608347199*pi) q[38];
U1q(0.761369685162492*pi,0.5760873382815901*pi) q[39];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[5],q[1];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[33],q[7];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[24],q[12];
RZZ(0.5*pi) q[32],q[13];
RZZ(0.5*pi) q[14],q[29];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[16],q[28];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[37],q[22];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[31],q[30];
RZZ(0.5*pi) q[35],q[38];
U1q(0.60452427168304*pi,0.6822665306476785*pi) q[0];
U1q(0.417985596502361*pi,0.9349203826173964*pi) q[1];
U1q(0.600768200815826*pi,0.43397567248803437*pi) q[2];
U1q(0.570560690476673*pi,0.7077994965309902*pi) q[3];
U1q(0.556547997047314*pi,1.1916273089542226*pi) q[4];
U1q(0.582690299024641*pi,0.24777985628245158*pi) q[5];
U1q(0.670324668585024*pi,0.8184279208723995*pi) q[6];
U1q(0.402457049027325*pi,1.69492792260134*pi) q[7];
U1q(0.563311362778677*pi,0.6244431356940501*pi) q[8];
U1q(0.509625647282196*pi,1.3414866521357798*pi) q[9];
U1q(0.952328084870579*pi,1.1116424026236897*pi) q[10];
U1q(0.348145449970112*pi,0.23402199818547587*pi) q[11];
U1q(0.249556658950738*pi,1.03770536778323*pi) q[12];
U1q(0.776492864071953*pi,1.8834515775528704*pi) q[13];
U1q(0.598728460771797*pi,0.9610381285460621*pi) q[14];
U1q(0.362618253847265*pi,1.2307972725224632*pi) q[15];
U1q(0.691051573078014*pi,1.3790824790480234*pi) q[16];
U1q(0.227880690118615*pi,0.2149110341636966*pi) q[17];
U1q(0.543897904584056*pi,0.33339429819769517*pi) q[18];
U1q(0.703331355558134*pi,0.98553648539884*pi) q[19];
U1q(0.797706175779795*pi,1.3913550595343902*pi) q[20];
U1q(0.22107863386248*pi,1.14731933929803*pi) q[21];
U1q(0.733316422694669*pi,0.2695213110266286*pi) q[22];
U1q(0.46739971361185*pi,1.7182784352527225*pi) q[23];
U1q(0.314469217353923*pi,1.3132164823706471*pi) q[24];
U1q(0.194154486401534*pi,0.6251306175173639*pi) q[25];
U1q(0.719134240890656*pi,1.10190313699111*pi) q[26];
U1q(0.727261789605283*pi,1.039091656829405*pi) q[27];
U1q(0.758975418909442*pi,1.26591039634615*pi) q[28];
U1q(0.50438355754682*pi,1.8838615768491591*pi) q[29];
U1q(0.835800464660447*pi,0.34965441723138646*pi) q[30];
U1q(0.31214112260299*pi,1.7947858095403753*pi) q[31];
U1q(0.547022857630195*pi,1.3784968132009698*pi) q[32];
U1q(0.252813840940705*pi,0.8735831039885298*pi) q[33];
U1q(0.0743448142178152*pi,0.9475784971887853*pi) q[34];
U1q(0.688328119983165*pi,0.9740558125943162*pi) q[35];
U1q(0.573620489228384*pi,1.3969237419398075*pi) q[36];
U1q(0.210401454470688*pi,0.2632858682736492*pi) q[37];
U1q(0.684509902575254*pi,1.1345663975499498*pi) q[38];
U1q(0.72491090353291*pi,1.9407155934027696*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[28],q[1];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[3],q[34];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[31];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[10],q[25];
RZZ(0.5*pi) q[27],q[11];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[15],q[14];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[19],q[30];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[23],q[29];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[26],q[36];
U1q(0.514644587988656*pi,0.8766283292999493*pi) q[0];
U1q(0.446208709479464*pi,0.11692288903388537*pi) q[1];
U1q(0.461493538721336*pi,1.131737022405214*pi) q[2];
U1q(0.588926666054847*pi,1.19860315323883*pi) q[3];
U1q(0.841079201524268*pi,1.4929021223512127*pi) q[4];
U1q(0.373242657026495*pi,0.5480505798708917*pi) q[5];
U1q(0.858403450328078*pi,1.19918822429646*pi) q[6];
U1q(0.744718926728969*pi,1.3716120646051504*pi) q[7];
U1q(0.136032837838648*pi,0.7081602924808799*pi) q[8];
U1q(0.34783922948372*pi,0.6939406226292499*pi) q[9];
U1q(0.290861150580263*pi,0.3628208896156302*pi) q[10];
U1q(0.816013502292399*pi,0.060679050159295755*pi) q[11];
U1q(0.708296206403862*pi,1.0151195763996101*pi) q[12];
U1q(0.505745323619917*pi,0.38267141053658005*pi) q[13];
U1q(0.677861986276768*pi,0.4493893372324429*pi) q[14];
U1q(0.748535235507421*pi,1.0312725700594934*pi) q[15];
U1q(0.222232509290475*pi,1.5728696740868129*pi) q[16];
U1q(0.305340766316176*pi,1.149326449930637*pi) q[17];
U1q(0.414580887506456*pi,0.6288715923322155*pi) q[18];
U1q(0.36942118222625*pi,1.6304082024539701*pi) q[19];
U1q(0.742670663577822*pi,0.060609944913160074*pi) q[20];
U1q(0.389258018943996*pi,1.3143824096504497*pi) q[21];
U1q(0.333821103314593*pi,1.9467880367873178*pi) q[22];
U1q(0.791532042997866*pi,1.9521376889259017*pi) q[23];
U1q(0.702089470533575*pi,1.3005843454961372*pi) q[24];
U1q(0.301861939899898*pi,1.4170220622799334*pi) q[25];
U1q(0.484756187440118*pi,0.16552878452096031*pi) q[26];
U1q(0.42087244074359*pi,0.5385334380669562*pi) q[27];
U1q(0.660719819687218*pi,1.5352088034882199*pi) q[28];
U1q(0.870180033536845*pi,0.38398396348476993*pi) q[29];
U1q(0.307927453811131*pi,0.30162277375109614*pi) q[30];
U1q(0.643429354179552*pi,1.4586467969471952*pi) q[31];
U1q(0.613704673044503*pi,0.6065144480027502*pi) q[32];
U1q(0.697602134163319*pi,0.3187802128860504*pi) q[33];
U1q(0.815866375428984*pi,1.1717480971028653*pi) q[34];
U1q(0.565854424776826*pi,0.33589714867938625*pi) q[35];
U1q(0.902119967672719*pi,0.41868704664493706*pi) q[36];
U1q(0.452083817503603*pi,1.088485894806749*pi) q[37];
U1q(0.428640555570861*pi,1.0819590818179403*pi) q[38];
U1q(0.344125832602965*pi,1.9920340547236304*pi) q[39];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[4],q[32];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[19],q[7];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[11],q[38];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[37],q[16];
RZZ(0.5*pi) q[18],q[25];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[35],q[26];
RZZ(0.5*pi) q[27],q[30];
RZZ(0.5*pi) q[34],q[36];
U1q(0.711141532302259*pi,1.0420997411786477*pi) q[0];
U1q(0.695899023722296*pi,1.7803667906899552*pi) q[1];
U1q(0.48586000862465*pi,1.9893278681791342*pi) q[2];
U1q(0.655535037727975*pi,1.7331671677887996*pi) q[3];
U1q(0.666726200918099*pi,0.6099000204125531*pi) q[4];
U1q(0.280682401999779*pi,0.8682874523895316*pi) q[5];
U1q(0.338496120066005*pi,0.2137003834227187*pi) q[6];
U1q(0.564678490791445*pi,1.5446414686401706*pi) q[7];
U1q(0.3563924408794*pi,1.1871392078647407*pi) q[8];
U1q(0.178705669202736*pi,0.6614222149184705*pi) q[9];
U1q(0.271831947061966*pi,1.0565269446518997*pi) q[10];
U1q(0.234674719022278*pi,1.9793181426024073*pi) q[11];
U1q(0.805564699693836*pi,0.9656121588042001*pi) q[12];
U1q(0.484924431527452*pi,1.3216860730065*pi) q[13];
U1q(0.511283281158545*pi,0.5126942624065123*pi) q[14];
U1q(0.821792391240548*pi,0.11271311427109332*pi) q[15];
U1q(0.718741132726722*pi,1.0465171696371929*pi) q[16];
U1q(0.271149725861776*pi,0.25127827896887567*pi) q[17];
U1q(0.482120520311548*pi,1.8512070694358052*pi) q[18];
U1q(0.452545531137194*pi,0.5711600850857996*pi) q[19];
U1q(0.637513562757016*pi,0.8097650912320802*pi) q[20];
U1q(0.246347284308883*pi,0.6926983465726098*pi) q[21];
U1q(0.314073389604254*pi,1.623136556127589*pi) q[22];
U1q(0.38326483596534*pi,0.9167649496254615*pi) q[23];
U1q(0.711574210659839*pi,0.7085822524798271*pi) q[24];
U1q(0.759254050858383*pi,1.7259889564847342*pi) q[25];
U1q(0.339433553004027*pi,1.5230103606927994*pi) q[26];
U1q(0.536569869668553*pi,1.006035599777615*pi) q[27];
U1q(0.552509617201571*pi,1.3973439645110703*pi) q[28];
U1q(0.881477963797433*pi,0.9608550617968596*pi) q[29];
U1q(0.359802942620997*pi,1.5683847223980152*pi) q[30];
U1q(0.72330375758686*pi,1.0884019985147955*pi) q[31];
U1q(0.606598229200352*pi,1.2795828917671095*pi) q[32];
U1q(0.573725917554887*pi,1.8926462803646995*pi) q[33];
U1q(0.419925411245689*pi,1.5826584715873349*pi) q[34];
U1q(0.859059647668866*pi,1.089845292799116*pi) q[35];
U1q(0.417097476839955*pi,1.759641942549118*pi) q[36];
U1q(0.281193083733503*pi,0.7703638497762473*pi) q[37];
U1q(0.825038336243067*pi,0.7071591464230007*pi) q[38];
U1q(0.233513604976082*pi,0.3724613588601402*pi) q[39];
rz(2.278900967544512*pi) q[0];
rz(1.8566017954005947*pi) q[1];
rz(2.539410614761766*pi) q[2];
rz(0.7186847565026007*pi) q[3];
rz(2.9237618322732777*pi) q[4];
rz(0.058597102635769005*pi) q[5];
rz(2.9229103365606814*pi) q[6];
rz(0.46261838060554084*pi) q[7];
rz(1.2901007424996003*pi) q[8];
rz(1.6439187421469796*pi) q[9];
rz(2.1492864739031994*pi) q[10];
rz(1.2340638184431238*pi) q[11];
rz(2.68672484515566*pi) q[12];
rz(0.03832008174070012*pi) q[13];
rz(0.26494527576473814*pi) q[14];
rz(1.9368097388605374*pi) q[15];
rz(2.1995154886113566*pi) q[16];
rz(2.672312224525884*pi) q[17];
rz(0.04297395878277399*pi) q[18];
rz(2.47699332453646*pi) q[19];
rz(1.5766992850517596*pi) q[20];
rz(3.4310981727513097*pi) q[21];
rz(1.3743679571164726*pi) q[22];
rz(1.498129769761709*pi) q[23];
rz(3.1926723497956724*pi) q[24];
rz(3.652192621757197*pi) q[25];
rz(2.0267228772148993*pi) q[26];
rz(2.464860012079985*pi) q[27];
rz(2.7177611571104103*pi) q[28];
rz(1.7251065642794003*pi) q[29];
rz(3.8586257602700043*pi) q[30];
rz(1.4941157926345436*pi) q[31];
rz(1.2813186000085093*pi) q[32];
rz(0.8386616034213006*pi) q[33];
rz(0.45883805553586576*pi) q[34];
rz(2.976597759839434*pi) q[35];
rz(3.778213209533523*pi) q[36];
rz(2.1098060291216623*pi) q[37];
rz(3.691358335618199*pi) q[38];
rz(0.54142641146138*pi) q[39];
U1q(1.71114153230226*pi,0.321000708723164*pi) q[0];
U1q(0.695899023722296*pi,0.63696858609055*pi) q[1];
U1q(0.48586000862465*pi,1.5287384829409039*pi) q[2];
U1q(1.65553503772797*pi,1.451851924291403*pi) q[3];
U1q(1.6667262009181*pi,0.533661852685825*pi) q[4];
U1q(0.280682401999779*pi,1.9268845550253009*pi) q[5];
U1q(0.338496120066005*pi,0.136610719983391*pi) q[6];
U1q(0.564678490791445*pi,1.00725984924571*pi) q[7];
U1q(0.3563924408794*pi,1.477239950364343*pi) q[8];
U1q(0.178705669202736*pi,1.305340957065456*pi) q[9];
U1q(0.271831947061966*pi,0.205813418555149*pi) q[10];
U1q(1.23467471902228*pi,0.213381961045531*pi) q[11];
U1q(1.80556469969384*pi,0.652337003959867*pi) q[12];
U1q(0.484924431527452*pi,0.360006154747254*pi) q[13];
U1q(0.511283281158545*pi,1.777639538171258*pi) q[14];
U1q(3.821792391240548*pi,1.04952285313164*pi) q[15];
U1q(0.718741132726722*pi,0.246032658248546*pi) q[16];
U1q(1.27114972586178*pi,1.9235905034947647*pi) q[17];
U1q(0.482120520311548*pi,0.89418102821858*pi) q[18];
U1q(0.452545531137194*pi,0.0481534096222573*pi) q[19];
U1q(1.63751356275702*pi,1.386464376283849*pi) q[20];
U1q(1.24634728430888*pi,1.123796519323922*pi) q[21];
U1q(1.31407338960425*pi,1.9975045132440559*pi) q[22];
U1q(1.38326483596534*pi,1.414894719387164*pi) q[23];
U1q(1.71157421065984*pi,0.901254602275492*pi) q[24];
U1q(1.75925405085838*pi,0.378181578241929*pi) q[25];
U1q(0.339433553004027*pi,0.54973323790774*pi) q[26];
U1q(0.536569869668553*pi,0.470895611857563*pi) q[27];
U1q(0.552509617201571*pi,1.11510512162148*pi) q[28];
U1q(0.881477963797433*pi,1.6859616260762689*pi) q[29];
U1q(1.359802942621*pi,0.427010482668024*pi) q[30];
U1q(0.72330375758686*pi,1.582517791149336*pi) q[31];
U1q(1.60659822920035*pi,1.56090149177562*pi) q[32];
U1q(0.573725917554887*pi,1.731307883786055*pi) q[33];
U1q(1.41992541124569*pi,1.041496527123164*pi) q[34];
U1q(1.85905964766887*pi,1.06644305263855*pi) q[35];
U1q(1.41709747683996*pi,0.537855152082641*pi) q[36];
U1q(1.2811930837335*pi,1.880169878897874*pi) q[37];
U1q(0.825038336243067*pi,1.39851748204124*pi) q[38];
U1q(1.23351360497608*pi,1.913887770321519*pi) q[39];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[4],q[32];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[19],q[7];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[11],q[38];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[14],q[21];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[37],q[16];
RZZ(0.5*pi) q[18],q[25];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[35],q[26];
RZZ(0.5*pi) q[27],q[30];
RZZ(0.5*pi) q[34],q[36];
U1q(3.485355412011344*pi,1.486472120601866*pi) q[0];
U1q(1.44620870947946*pi,0.97352468443448*pi) q[1];
U1q(1.46149353872134*pi,0.671147637166979*pi) q[2];
U1q(3.411073333945152*pi,0.9864159388413758*pi) q[3];
U1q(1.84107920152427*pi,0.6506597507471733*pi) q[4];
U1q(0.373242657026495*pi,0.60664768250666*pi) q[5];
U1q(0.858403450328078*pi,0.12209856085712989*pi) q[6];
U1q(3.744718926728969*pi,1.83423044521069*pi) q[7];
U1q(1.13603283783865*pi,1.9982610349804801*pi) q[8];
U1q(0.34783922948372*pi,0.33785936477623*pi) q[9];
U1q(3.290861150580263*pi,0.51210736351884*pi) q[10];
U1q(3.1839864977076*pi,1.1320210534886366*pi) q[11];
U1q(3.291703793596138*pi,1.6028295863644566*pi) q[12];
U1q(3.5057453236199168*pi,1.420991492277327*pi) q[13];
U1q(1.67786198627677*pi,1.7143346129971802*pi) q[14];
U1q(1.74853523550742*pi,1.1309633973432371*pi) q[15];
U1q(0.222232509290475*pi,0.7723851626981599*pi) q[16];
U1q(1.30534076631618*pi,1.0255423325330077*pi) q[17];
U1q(1.41458088750646*pi,1.67184555111499*pi) q[18];
U1q(0.36942118222625*pi,0.10740152699044003*pi) q[19];
U1q(1.74267066357782*pi,1.1356195226027779*pi) q[20];
U1q(3.610741981056004*pi,1.502112456246076*pi) q[21];
U1q(3.666178896685407*pi,1.6738530325843173*pi) q[22];
U1q(1.79153204299787*pi,0.37952198008672156*pi) q[23];
U1q(3.702089470533575*pi,0.30925250925917736*pi) q[24];
U1q(3.698138060100101*pi,1.6871484724467272*pi) q[25];
U1q(0.484756187440118*pi,1.1922516617359*pi) q[26];
U1q(1.42087244074359*pi,0.0033934501468999567*pi) q[27];
U1q(0.660719819687218*pi,1.252969960598627*pi) q[28];
U1q(3.870180033536846*pi,0.10909052776418005*pi) q[29];
U1q(3.692072546188869*pi,0.6937724313149518*pi) q[30];
U1q(0.643429354179552*pi,1.9527625895817402*pi) q[31];
U1q(1.6137046730445*pi,1.2339699355399794*pi) q[32];
U1q(0.697602134163319*pi,0.15744181630738008*pi) q[33];
U1q(1.81586637542898*pi,0.45240690160759334*pi) q[34];
U1q(3.434145575223174*pi,0.8203911967582779*pi) q[35];
U1q(3.09788003232728*pi,1.8788100479868235*pi) q[36];
U1q(3.452083817503603*pi,1.5620478338673747*pi) q[37];
U1q(0.428640555570861*pi,1.77331741743614*pi) q[38];
U1q(1.34412583260297*pi,1.294315074458027*pi) q[39];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[28],q[1];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[3],q[34];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[31];
RZZ(0.5*pi) q[8],q[16];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[10],q[25];
RZZ(0.5*pi) q[27],q[11];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[15],q[14];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[19],q[30];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[23],q[29];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[26],q[36];
U1q(1.60452427168304*pi,0.6808339192541344*pi) q[0];
U1q(1.41798559650236*pi,0.155527190850971*pi) q[1];
U1q(3.399231799184174*pi,0.36890898708415876*pi) q[2];
U1q(1.57056069047667*pi,0.47721959554921733*pi) q[3];
U1q(1.55654799704731*pi,0.3493849373501833*pi) q[4];
U1q(1.58269029902464*pi,1.3063769589182197*pi) q[5];
U1q(1.67032466858503*pi,1.7413382574330698*pi) q[6];
U1q(1.40245704902733*pi,1.5109145872144953*pi) q[7];
U1q(3.436688637221323*pi,1.0819781917673192*pi) q[8];
U1q(0.509625647282196*pi,1.98540539428277*pi) q[9];
U1q(1.95232808487058*pi,0.7632858505107796*pi) q[10];
U1q(1.34814544997011*pi,0.9586781054624613*pi) q[11];
U1q(3.249556658950738*pi,1.5802437949808423*pi) q[12];
U1q(3.223507135928047*pi,1.9202113252610413*pi) q[13];
U1q(3.401271539228203*pi,0.20268582168355942*pi) q[14];
U1q(0.362618253847265*pi,1.3304880998061999*pi) q[15];
U1q(1.69105157307801*pi,0.57859796765938*pi) q[16];
U1q(1.22788069011861*pi,1.091126916766063*pi) q[17];
U1q(3.543897904584056*pi,0.9673228452495151*pi) q[18];
U1q(0.703331355558134*pi,1.462529809935302*pi) q[19];
U1q(1.7977061757798*pi,0.4663646372240109*pi) q[20];
U1q(3.77892136613752*pi,1.6691755265985062*pi) q[21];
U1q(3.266683577305331*pi,1.3511197583450074*pi) q[22];
U1q(0.46739971361185*pi,0.1456627264135415*pi) q[23];
U1q(1.31446921735392*pi,1.3218846461336824*pi) q[24];
U1q(3.805845513598466*pi,1.479039917209299*pi) q[25];
U1q(1.71913424089066*pi,0.12862601420605024*pi) q[26];
U1q(3.272738210394717*pi,1.5028352313844588*pi) q[27];
U1q(0.758975418909442*pi,0.9836715534565601*pi) q[28];
U1q(1.50438355754682*pi,0.609212914399788*pi) q[29];
U1q(1.83580046466045*pi,1.6457407878346615*pi) q[30];
U1q(1.31214112260299*pi,0.28890160217491*pi) q[31];
U1q(0.547022857630195*pi,1.0059523007381994*pi) q[32];
U1q(0.252813840940705*pi,1.7122447074098703*pi) q[33];
U1q(0.0743448142178152*pi,0.22823730169352352*pi) q[34];
U1q(3.311671880016834*pi,1.1822325328433578*pi) q[35];
U1q(3.426379510771615*pi,0.9005733526919473*pi) q[36];
U1q(0.210401454470688*pi,1.7368478073342706*pi) q[37];
U1q(0.684509902575254*pi,0.8259247331681401*pi) q[38];
U1q(0.72491090353291*pi,1.242996613137167*pi) q[39];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[5],q[1];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[33],q[7];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[24],q[12];
RZZ(0.5*pi) q[32],q[13];
RZZ(0.5*pi) q[14],q[29];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[16],q[28];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[37],q[22];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[31],q[30];
RZZ(0.5*pi) q[35],q[38];
U1q(0.29614156020767*pi,1.8462300705439842*pi) q[0];
U1q(0.326215791658225*pi,0.9527386207363311*pi) q[1];
U1q(3.503902410775646*pi,1.8720570309484026*pi) q[2];
U1q(0.689946932281283*pi,0.7615400969741075*pi) q[3];
U1q(3.52918735031052*pi,0.6492144490114592*pi) q[4];
U1q(3.435341783501306*pi,1.052661415506709*pi) q[5];
U1q(1.38471810565108*pi,0.9406107917898998*pi) q[6];
U1q(1.06882518771906*pi,0.6670918499660752*pi) q[7];
U1q(3.350442100831763*pi,1.145325843624839*pi) q[8];
U1q(3.5683899845814198*pi,0.7137216688857304*pi) q[9];
U1q(0.447980605111878*pi,0.9436678587940097*pi) q[10];
U1q(1.61980416821414*pi,1.3934256029068615*pi) q[11];
U1q(1.37824898575287*pi,1.458253437290061*pi) q[12];
U1q(1.0847288239837*pi,0.004956406258979085*pi) q[13];
U1q(3.561478056296851*pi,0.22309277607819955*pi) q[14];
U1q(1.38982348033695*pi,0.7501654152697301*pi) q[15];
U1q(3.894531646364699*pi,1.8718157927697903*pi) q[16];
U1q(1.4742779015606*pi,1.2696837065396285*pi) q[17];
U1q(0.648139454694005*pi,0.08512624100211497*pi) q[18];
U1q(1.84547734998025*pi,1.4877181298130409*pi) q[19];
U1q(1.53927546758859*pi,0.3890850652690537*pi) q[20];
U1q(3.376960964550961*pi,0.6307153220355457*pi) q[21];
U1q(3.886018035037353*pi,0.9513066061573183*pi) q[22];
U1q(1.50800117394311*pi,0.1611884721903376*pi) q[23];
U1q(3.746002460993178*pi,1.9987745284001517*pi) q[24];
U1q(3.811821966757457*pi,1.110655204857129*pi) q[25];
U1q(1.81437683301186*pi,0.9342452906251495*pi) q[26];
U1q(1.53663085552682*pi,0.727207788017683*pi) q[27];
U1q(1.60882010479051*pi,0.67467395725525*pi) q[28];
U1q(1.50151372913291*pi,0.015592592207127831*pi) q[29];
U1q(0.495821930062629*pi,0.3558261384571715*pi) q[30];
U1q(1.86504365098748*pi,1.2259444259526546*pi) q[31];
U1q(3.145522486024939*pi,0.5016037496963381*pi) q[32];
U1q(0.10071673522969*pi,1.7597681659579703*pi) q[33];
U1q(0.340734404933302*pi,1.207998033698253*pi) q[34];
U1q(3.680799636339927*pi,1.016281334328188*pi) q[35];
U1q(3.252994102526987*pi,0.775276848958083*pi) q[36];
U1q(3.289986196257993*pi,1.6421041795252505*pi) q[37];
U1q(0.283661430432589*pi,1.5733709964529101*pi) q[38];
U1q(1.76136968516249*pi,0.8783683580159869*pi) q[39];
RZZ(0.5*pi) q[0],q[38];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[2],q[25];
RZZ(0.5*pi) q[3],q[26];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[9],q[37];
RZZ(0.5*pi) q[22],q[10];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[24],q[39];
RZZ(0.5*pi) q[27],q[32];
RZZ(0.5*pi) q[28],q[34];
U1q(0.620676880643457*pi,0.35017195220477415*pi) q[0];
U1q(0.87498564563381*pi,1.7303076850578512*pi) q[1];
U1q(1.65664470503468*pi,1.0428108833033163*pi) q[2];
U1q(0.801938737125873*pi,1.749721892030677*pi) q[3];
U1q(0.471563828240787*pi,0.25956279213606903*pi) q[4];
U1q(1.61609321619081*pi,1.356173933630469*pi) q[5];
U1q(0.330380497503687*pi,0.7538509630974506*pi) q[6];
U1q(1.72067499830151*pi,1.9225572250847862*pi) q[7];
U1q(1.21553298413*pi,0.13402035023687864*pi) q[8];
U1q(1.10110055706813*pi,1.0898177370058493*pi) q[9];
U1q(0.259767938612748*pi,1.5858847669919198*pi) q[10];
U1q(1.38623502317946*pi,0.7022150647214191*pi) q[11];
U1q(1.12677513512532*pi,0.29775696042045485*pi) q[12];
U1q(0.384974224745688*pi,0.47777098589495903*pi) q[13];
U1q(3.484997374228572*pi,1.556581277506587*pi) q[14];
U1q(1.30373460900054*pi,0.5836205410318012*pi) q[15];
U1q(1.79054062003504*pi,1.9547275950352339*pi) q[16];
U1q(0.7064758503465*pi,0.35476046131824823*pi) q[17];
U1q(0.44014135792224*pi,0.8653327478379156*pi) q[18];
U1q(1.56590988350276*pi,0.19100986758392907*pi) q[19];
U1q(0.526885386125895*pi,0.7375146745359566*pi) q[20];
U1q(1.52076551202626*pi,0.2599163140323588*pi) q[21];
U1q(1.2605663083892*pi,0.5025676624595636*pi) q[22];
U1q(1.73916870839652*pi,0.42405335233437835*pi) q[23];
U1q(0.248635304275606*pi,0.004504995068121076*pi) q[24];
U1q(1.77237673123335*pi,1.7183370768139405*pi) q[25];
U1q(0.648599981209124*pi,1.4876340822485297*pi) q[26];
U1q(0.75659472223703*pi,0.09629947014131313*pi) q[27];
U1q(3.148523813423686*pi,1.3306616210713837*pi) q[28];
U1q(1.24871783219827*pi,1.8773846584390292*pi) q[29];
U1q(0.520987685097743*pi,1.1730044391044716*pi) q[30];
U1q(0.389808260385542*pi,0.5300829637431042*pi) q[31];
U1q(1.59782405609277*pi,1.1799553676852628*pi) q[32];
U1q(0.722065998063701*pi,1.3473271108865692*pi) q[33];
U1q(0.837317869894606*pi,0.820872825513173*pi) q[34];
U1q(1.33305187356399*pi,1.8029306475764129*pi) q[35];
U1q(0.47335069412627*pi,0.300860613798851*pi) q[36];
U1q(1.45384319890066*pi,0.33608494897823693*pi) q[37];
U1q(0.423587663986964*pi,0.12798385115763988*pi) q[38];
U1q(1.73520645555331*pi,1.0698110226266966*pi) q[39];
rz(3.649828047795226*pi) q[0];
rz(2.269692314942149*pi) q[1];
rz(0.9571891166966837*pi) q[2];
rz(0.25027810796932304*pi) q[3];
rz(1.740437207863931*pi) q[4];
rz(0.643826066369531*pi) q[5];
rz(1.2461490369025494*pi) q[6];
rz(0.07744277491521379*pi) q[7];
rz(1.8659796497631214*pi) q[8];
rz(2.9101822629941507*pi) q[9];
rz(2.41411523300808*pi) q[10];
rz(3.297784935278581*pi) q[11];
rz(1.7022430395795451*pi) q[12];
rz(3.522229014105041*pi) q[13];
rz(2.443418722493413*pi) q[14];
rz(1.4163794589681988*pi) q[15];
rz(0.0452724049647662*pi) q[16];
rz(3.6452395386817518*pi) q[17];
rz(1.1346672521620844*pi) q[18];
rz(3.808990132416071*pi) q[19];
rz(1.2624853254640434*pi) q[20];
rz(3.740083685967641*pi) q[21];
rz(1.4974323375404364*pi) q[22];
rz(3.5759466476656216*pi) q[23];
rz(1.995495004931879*pi) q[24];
rz(2.2816629231860595*pi) q[25];
rz(0.5123659177514703*pi) q[26];
rz(1.9037005298586869*pi) q[27];
rz(2.6693383789286163*pi) q[28];
rz(2.1226153415609708*pi) q[29];
rz(2.8269955608955284*pi) q[30];
rz(3.469917036256896*pi) q[31];
rz(2.8200446323147372*pi) q[32];
rz(0.6526728891134308*pi) q[33];
rz(3.179127174486827*pi) q[34];
rz(0.19706935242358714*pi) q[35];
rz(1.699139386201149*pi) q[36];
rz(3.663915051021763*pi) q[37];
rz(3.87201614884236*pi) q[38];
rz(0.9301889773733035*pi) q[39];
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