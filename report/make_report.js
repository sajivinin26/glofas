const fs=require('fs'); const path=require('path');
const {Document,Packer,Paragraph,TextRun,Table,TableRow,TableCell,ImageRun,HeadingLevel,AlignmentType,WidthType,ShadingType,BorderStyle,LevelFormat,Footer,PageNumber,PageBreak}=require('docx');
const ROOT=path.resolve(__dirname,'..'); const R=JSON.parse(fs.readFileSync(ROOT+'/outputs/results.json')); const X=JSON.parse(fs.readFileSync(ROOT+'/outputs/xai_summary.json'));
const W=9638, f3=v=>v.toFixed(3), f2=v=>v.toFixed(2);
const P=(t,o={})=>new Paragraph({spacing:{after:120,line:276},alignment:o.align,keepNext:o.keepNext,children:(Array.isArray(t)?t:[t]).map(x=>typeof x==='string'?new TextRun({text:x,italics:o.i,bold:o.b,size:o.size}):x)});
const B=t=>new Paragraph({numbering:{reference:'bul',level:0},spacing:{after:60,line:276},children:(Array.isArray(t)?t:[t]).map(x=>typeof x==='string'?new TextRun(x):x)});
const b=t=>new TextRun({text:t,bold:true}); const H1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,children:[new TextRun(t)]}); const H2=t=>new Paragraph({heading:HeadingLevel.HEADING_2,children:[new TextRun(t)]});
const cap=t=>new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:200},children:[new TextRun({text:t,italics:true,size:19})]});
function img(file,wDxaCm){const buf=fs.readFileSync(ROOT+'/figures/'+file);const w=buf.readUInt32BE(16),h=buf.readUInt32BE(20);const pw=Math.round(wDxaCm*37.8);return new Paragraph({alignment:AlignmentType.CENTER,keepNext:true,spacing:{before:120,after:60},children:[new ImageRun({type:'png',data:buf,transformation:{width:pw,height:Math.round(pw*h/w)},altText:{title:file,description:file,name:file}})]});}
const bd={style:BorderStyle.SINGLE,size:4,color:'BBBBBB'}; const borders={top:bd,bottom:bd,left:bd,right:bd};
function table(cols,head,rows,opts={}){const sum=cols.reduce((a,c)=>a+c,0);const cell=(t,i,hd,bold)=>new TableCell({borders,width:{size:cols[i],type:WidthType.DXA},shading:hd?{fill:'D9E2F3',type:ShadingType.CLEAR,color:'auto'}:(bold?{fill:'F2F2F2',type:ShadingType.CLEAR,color:'auto'}:undefined),margins:{top:50,bottom:50,left:90,right:90},children:[new Paragraph({alignment:i>0&&!opts.left?AlignmentType.CENTER:AlignmentType.LEFT,children:[new TextRun({text:String(t),bold:hd||bold,size:opts.size||19})]})]});
 return new Table({width:{size:sum,type:WidthType.DXA},columnWidths:cols,rows:[new TableRow({tableHeader:true,children:head.map((h,i)=>cell(h,i,true))}),...rows.map(r=>new TableRow({cantSplit:true,children:r.map((t,i)=>cell(t,i,false,opts.boldRow&&opts.boldRow(r)))}))]});}
const nm=s=>s.replace(' (lecture baseline)',' *').replace(' (gradient boosting)','').replace(' (subsampled)','').replace('Rule: flow >= x * high-flow level (baseline)','Rule: flow/level ≥ 0.5 *');
const res=R.results; const lgb=res.find(r=>r.model.startsWith('LightGBM')); const lr=res.find(r=>r.model.startsWith('Logistic')); const ebm=res.find(r=>r.model.startsWith('Explainable')); const gp=res.find(r=>r.model.startsWith('Gaussian'));
const C=R.counts, rate=k=>(100*C[k][1]/C[k][0]).toFixed(1);
const kids=[];
kids.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:1200,after:200},children:[new TextRun({text:'Applied Machine Learning',size:26,color:'555555'})]}));
kids.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:200},children:[new TextRun({text:'Three-Day Flood-Onset Early Warning for Sri Lankan Rivers',bold:true,size:44})]}));
kids.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:600},children:[new TextRun({text:'Comparing LightGBM, Explainable Boosting Machine and Gaussian Process models on GloFAS river-discharge data, with SHAP explanations',size:26,color:'555555'})]}));
kids.push(P([b('Student: '),'[Your name]   ',b('Student ID: '),'[Your ID]'],{align:AlignmentType.CENTER}));
kids.push(P([b('Course assignment: '),'Machine Learning Assignment (MSc in AI)'],{align:AlignmentType.CENTER}));
kids.push(new Paragraph({children:[new PageBreak()]}));

kids.push(H1('Summary'));
kids.push(P(`River flooding is a recurring hazard in Sri Lanka. This project asks a practical question: on a day when a river is still below its usual high-flow level, can a machine-learning model tell, using only the river's own recent flow history, whether the flow will cross that level within the next three days? I built a table of daily flow for ${33} river points from the Copernicus GloFAS v4 reanalysis (2 January 2024 to 25 October 2026) and compared two lecture baselines (logistic regression, random forest) with three algorithms not taught in lectures: LightGBM gradient boosting, an Explainable Boosting Machine (EBM) and a Gaussian Process classifier.`));
kids.push(P(`Models were tuned on a validation period and evaluated once on a later test period (January to October 2026). LightGBM was the best of the new algorithms on validation (PR-AUC ${f3(lgb.valid.pr_auc)}) and reached a test PR-AUC of ${f3(lgb.test.pr_auc)} against a no-skill level of ${(C.test[1]/C.test[0]).toFixed(3)}. However, the simple logistic regression baseline was close behind (test PR-AUC ${f3(lr.test.pr_auc)}) and scored higher on test F1. With only ${C.test[1]} onset events in the test period, concentrated in about two episodes, the evidence that the newer algorithms beat the baseline is weak. SHAP and partial-dependence analysis show that the model relies mainly on how close today's flow is to the high-flow level and on how fast the flow is rising, which agrees with basic hydrological reasoning. The model is a learning exercise built on simulated data, not an operational warning system.`));

kids.push(H1('1. Problem definition and dataset'));
kids.push(H2('1.1 Problem and relevance'));
kids.push(P('Floods caused by the south-west and north-east monsoons and inter-monsoon storms affect many river basins in Sri Lanka. An early warning is useful only if it comes before the river crosses a damaging level. I therefore frame the task as flood onset: using days on which the flow is currently below the river\'s high-flow level, predict whether it will reach or exceed that level on any of the next three days. Days that are already above the level are excluded, because predicting "still high" is trivial and is not an early warning.'));
kids.push(H2('1.2 Data source'));
kids.push(P('The data are the GloFAS v4 historical reanalysis (LISFLOOD hydrological model driven by ERA5 weather), published by the Copernicus Emergency Management Service. I downloaded daily river discharge ("river discharge in the last 24 hours", m³/s) from the Copernicus Early Warning Data Store (EWDS) for a box around Sri Lanka (North 10.0, West 79.4, South 5.8, East 82.0) at 0.05° (about 5 km) resolution, one year per request because of the service\'s request-size limit. Use is under the CEMS-FLOODS datasets licence. The data contain no personal or sensitive information.'));
kids.push(P('Important: GloFAS reanalysis is a model simulation, not a record of gauge readings. In this report "flow" always means simulated flow. The most recent months of 2026 may come from a provisional version of the product; I did not verify this.'));
kids.push(H2('1.3 Dataset size, features and target'));
kids.push(table([3000,6638],['Item','Description'],[
 ['Period','2 Jan 2024 to 25 Oct 2026, 1,027 daily values per grid cell (one missing day, 1 Jan 2025, filled by linear interpolation)'],
 ['River points','33 grid cells with mean flow of at least 25 m³/s, chosen greedily from the largest rivers and kept at least 0.15° apart (mean flow 25 to 218 m³/s). Points are identified by grid position and are not matched to river names.'],
 ['High-flow level','90th percentile of each point\'s own flow in the training period (61 to 636 m³/s across points)'],
 ['Target y','1 if the maximum flow over the next 3 days is at or above the high-flow level, else 0. Only days currently below the level are kept.'],
 ['Rows','Train '+C.train[0].toLocaleString()+' ('+C.train[1]+' positives, '+rate('train')+'%), validation '+C.valid[0].toLocaleString()+' ('+C.valid[1]+', '+rate('valid')+'%), test '+C.test[0].toLocaleString()+' ('+C.test[1]+', '+rate('test')+'%)'],
 ['Features (11)','Flow today, 1, 2, 3 and 7 days ago, 7-day mean, 7-day maximum and 30-day mean, each divided by the point\'s high-flow level; 1-day and 3-day relative rise; flow today divided by 30-day mean']
],{left:true}));
kids.push(P('',{}));
kids.push(H2('1.4 Preprocessing and ethical use'));
kids.push(B('Three NetCDF files (one was uploaded twice and the duplicate removed) were concatenated, sorted by date and checked for gaps; the single missing day was interpolated.'));
kids.push(B('Every flow feature is divided by the point\'s high-flow level so that the model sees how close a river is to its own threshold rather than the absolute size of the river. This also lets one model serve many rivers.'));
kids.push(B('Seasonal (day-of-year) features were deliberately left out. With less than two years of training data, the calendar day acts as a date identifier and tree models memorise individual storms. In a quick check, a random forest with seasonal features scored a test PR-AUC of 0.03, compared with 0.20 without them. I found this after seeing test results, so test scores for tree models are slightly optimistic; this is acknowledged in Section 5.'));
kids.push(B('The data are public and contain nothing about individuals. The model outputs are not suitable for real warnings or evacuation decisions.'));

kids.push(H1('2. Selection of new algorithms'));
kids.push(P('The lecture models (linear and logistic regression, SVM, Naive Bayes, bagging and random forests, clustering) are used only as baselines. Three algorithms outside the lecture content were compared, none of them deep learning.'));
kids.push(table([2300,3700,3638],['Algorithm','How it works','Why it was chosen and how it differs'],[
 ['LightGBM (gradient boosting)','Builds many small trees one after another; each new tree fits the errors of the ensemble so far. Uses histogram-based, leaf-wise tree growth.','Strong on tabular data with rare positives. Unlike a random forest, whose trees are independent and averaged, boosting corrects mistakes sequentially and can reduce bias as well as variance.'],
 ['Explainable Boosting Machine (EBM)','A generalised additive model fitted by cyclic boosting: one smooth shape function per feature (plus a few pairwise terms). The prediction is the sum of the shape functions.','Accuracy close to boosted trees but intrinsically interpretable, so each feature\'s effect can be read directly. Differs from logistic regression because each effect can be non-linear.'],
 ['Gaussian Process classifier','Bayesian, kernel-based model that places a prior over functions and gives a probabilistic prediction.','Brings a probabilistic, similarity-based view different from trees and from the SVM kernel idea in lectures. Cost grows roughly with n³, so it was trained on a subsample.']
],{left:true,size:18}));
kids.push(P('',{}));
kids.push(P('Baselines for context: (a) a simple rule that raises an alert when today\'s flow is above a fraction of the high-flow level, (b) logistic regression with class weights, (c) random forest with class weights. In the tables below, * marks baselines.'));

kids.push(H1('3. Model training and evaluation'));
kids.push(H2('3.1 Time-based split'));
kids.push(P('Daily river flow is strongly autocorrelated, so a random split would place almost identical days in both training and test sets. The data were therefore split by time:'));
kids.push(B([b('Training: '),'31 Jan 2024 to 30 Jun 2025 ('+C.train[0].toLocaleString()+' rows).']));
kids.push(B([b('Validation: '),'1 Jul to 31 Dec 2025 ('+C.valid[0].toLocaleString()+' rows), used to tune hyperparameters, choose the alert threshold and choose the best new algorithm.']));
kids.push(B([b('Test: '),'1 Jan to 22 Oct 2026 ('+C.test[0].toLocaleString()+' rows), used once for final reporting. The last three days have no known outcome and are excluded.']));
kids.push(H2('3.2 Hyperparameters and metrics'));
kids.push(P('Each model was tuned with a small grid, scored by validation PR-AUC (average precision). Positives are rare (1 to 3% of rows), so accuracy would be misleading; I report PR-AUC, ROC-AUC, precision, recall, F1 and MCC. The alert threshold for each model was chosen to maximise F1 on the validation set and then fixed for the test set. Class imbalance was handled with class weights (logistic regression, random forest, LightGBM scale_pos_weight) and a stratified subsample of negatives for the Gaussian Process (all 352 positives plus 2,000 negatives).'));
kids.push(table([3300,6338],['Model','Selected hyperparameters (grid searched on validation)'],[
 ['Logistic regression *','C = 10 (grid 0.01, 0.1, 1, 10), standardised inputs'],
 ['Random forest *','300 trees, max_features 0.3, min_samples_leaf 10'],
 ['LightGBM','150 trees, learning rate 0.03, 8 leaves, min_child_samples 30, subsample 0.8, colsample 0.8 (8 combinations tried)'],
 ['EBM','learning rate 0.03, max_leaves 3, no pairwise interactions (3 configurations tried)'],
 ['Gaussian Process','RBF kernel with constant scale, length scale 1.0 (1.0 and 3.0 tried), standardised inputs']
],{left:true}));
kids.push(P('',{}));
kids.push(H2('3.3 Results'));
const rows=res.map(r=>[nm(r.model),f3(r.valid.pr_auc),f3(r.test.pr_auc),f3(r.test.roc_auc),f3(r.test.precision),f3(r.test.recall),f3(r.test.f1),f3(r.test.mcc)]);
kids.push(table([2900,1000,1000,1000,1000,900,900,938],['Model','Valid PR-AUC','Test PR-AUC','Test ROC-AUC','Test precision','Test recall','Test F1','Test MCC'],rows,{size:17,left:false,boldRow:r=>r[0].startsWith('LightGBM')}));
kids.push(cap('Table 1. Model comparison. * lecture baseline. No-skill test PR-AUC = 0.010. LightGBM (shaded) is the selected model.'));
kids.push(img('model_comparison.png',12)); kids.push(cap('Figure 1. Test PR-AUC by model.'));
kids.push(img('pr_curves.png',12.5)); kids.push(cap('Figure 2. Precision-recall curves on the 2026 test period.'));
kids.push(H2('3.4 What the results indicate'));
kids.push(B([b('Model selection. '),`Among the new algorithms LightGBM had the best validation PR-AUC (${f3(lgb.valid.pr_auc)}), ahead of EBM (${f3(ebm.valid.pr_auc)}) and the Gaussian Process (${f3(gp.valid.pr_auc)}), so LightGBM was selected before looking at the test set.`]));
kids.push(B([b('Selected model on test. '),`With the validation-chosen threshold of ${f2(lgb.thr)}, LightGBM found ${lgb.test.tp} of the ${C.test[1]} onsets (recall ${f2(lgb.test.recall)}) but raised ${lgb.test.fp} false alarms (precision ${f2(lgb.test.precision)}). ROC-AUC was ${f2(lgb.test.roc_auc)}. The test PR-AUC of ${f3(lgb.test.pr_auc)} is about 27 times the no-skill level of 0.010, so the model carries real signal.`]));
kids.push(B([b('Honest comparison with the baselines. '),`Logistic regression had the highest validation PR-AUC overall (${f3(lr.valid.pr_auc)}) and the best test F1 (${f3(lr.test.f1)}) and MCC (${f3(lr.test.mcc)}); LightGBM had the best test PR-AUC. The new algorithms do not clearly beat the simple baseline. With ${C.test[1]} test positives (66 in October 2026 and 18 in May 2026), these differences are well within what chance could produce.`]));
kids.push(B([b('Simple rule. '),`The rule "alert when flow is at least half the high-flow level" has recall of ${f2(res[0].test.recall)} but precision of only ${f2(res[0].test.precision)}, so the machine-learning models mostly improve precision.`]));
kids.push(B([b('Gaussian Process. '),'It performed worst, probably because it saw only about 2,350 of the 15,559 training rows and a single isotropic length scale.']));
kids.push(B([b('Probabilities. '),'Class weighting inflates predicted probabilities, so they should be read as risk scores, not calibrated probabilities.']));

kids.push(H1('4. Explainability and interpretation'));
kids.push(P('The selected LightGBM model was explained with SHAP (TreeExplainer, on the test period) and partial dependence plots. As a cross-check, the importance of each term in the EBM, which is interpretable by construction, was compared with the SHAP ranking. SHAP values are in log-odds units: positive values push towards "high flow within 3 days".'));
kids.push(img('shap_beeswarm.png',12)); kids.push(cap('Figure 3. SHAP summary plot (each dot is one test day; colour is the feature value).'));
kids.push(H2('4.1 Which features matter most'));
const top=Object.entries(X.top_shap);
kids.push(table([5200,2200],['Feature','Mean |SHAP|'],top.map(([k,v])=>[k,f3(v)]),{left:true}));
kids.push(cap('Table 2. Five most influential features by mean absolute SHAP value.'));
kids.push(B([b('Closeness to the level dominates. '),'High values of flow today divided by the high-flow level push the prediction strongly upwards (up to about +3.8 log-odds), and low values push it down.']));
kids.push(B([b('Momentum matters. '),'A high flow today compared with the 30-day mean, and a positive 1-day or 3-day relative rise, raise the risk: the river is climbing, not just high.']));
kids.push(B([b('Recession lowers risk. '),'Where flow 3 or 7 days ago was high relative to the level (a falling river), SHAP values are often negative.']));
kids.push(img('pdp_top4.png',12)); kids.push(cap('Figure 4. Partial dependence for the four most important features.'));
kids.push(P('The partial dependence plots show the average effect of each feature. The risk score rises roughly steadily with flow relative to the level; it jumps once flow exceeds about 0.6 to 0.8 times its 30-day mean; and it jumps once the 1-day change is no longer negative, then stays flat. The small wiggles in the curves reflect the limited data and should not be over-interpreted.'));
kids.push(img('shap_waterfall.png',11)); kids.push(cap(`Figure 5. Local explanation for one correctly detected onset (point ${X.local_case.point}, ${X.local_case.date}, risk score ${f2(X.local_case.p)}).`));
kids.push(P('In this case the flow was already about 0.89 of the high-flow level, rose 68% in one day and 38% over three days, and was 3.3 times its 30-day mean. All of these contribute positively, which is the situation in which a forecaster would also be concerned.'));
kids.push(H2('4.2 Does the model match domain knowledge?'));
kids.push(P(`Largely yes. A river close to its flood level and rising quickly is the classic precursor of a threshold crossing, and a falling river is less likely to cross. The model learned this without any hydrological rules. Two cautions: (1) the EBM ranks the features differently (its top terms are the 30-day mean, flow 3 days ago, 3-day rise and flow today), and the rank correlation between the SHAP and EBM importances is only ${f2(X.spearman_shap_ebm)}. The features are strongly correlated with each other, so importance can be shared among them in different ways, and the exact ordering should not be taken as a firm finding. (2) The explanations describe the model, not the physical river system, and are not causal: the model has no information on rainfall.`));

kids.push(H1('5. Critical discussion'));
kids.push(H2('5.1 Limitations of the model'));
kids.push(B('Many false alarms: at the chosen threshold roughly 85% of alerts are wrong (precision about 0.15). A real system would need a threshold tuned to the cost of a missed flood versus the cost of a false alarm.'));
kids.push(B('No rainfall input. The model sees only past flow, so it cannot anticipate a flood caused by rain that has not yet reached the river. Adding ERA5 or forecast rainfall would likely help most.'));
kids.push(B('Short record. About 2.8 years (17 months of training) contain only a few flood episodes. Test positives come mainly from two events, so the test score is noisy and may not generalise to other seasons or years.'));
kids.push(B('The high-flow level is a statistical percentile of the first 17 months, not an official flood stage or a GloFAS return-period threshold.'));
kids.push(B('Model selection used test information once: seasonal features were dropped after poor test results for tree models, so their test scores are slightly optimistic.'));
kids.push(B('Probabilities are not calibrated and one model is pooled across 33 points.'));
kids.push(H2('5.2 Data quality issues'));
kids.push(B('GloFAS is a simulation driven by ERA5 and is known to be weakest on short, steep catchments such as the wet-zone and hill-country rivers. Its errors become the model\'s "truth".'));
kids.push(B('0.05° cells are coarse; a cell can contain several small streams. The points are not matched to named rivers or gauges, so results cannot be checked against Irrigation Department records.'));
kids.push(B('The newest months may come from a provisional GloFAS product, and one day (1 Jan 2025) is missing and was interpolated.'));
kids.push(H2('5.3 Bias, fairness and ethics'));
kids.push(B('Spatial unfairness: performance probably differs between basins and between the wet and dry zones. Communities in basins where the model is weak would get worse warnings. Per-basin evaluation should be done before any use.'));
kids.push(B('Larger, well-resolved rivers are over-represented because points were selected by mean flow, so small flashy catchments, which are often the most dangerous, are not covered.'));
kids.push(B('Alert fatigue and trust: frequent false alarms can make people ignore warnings; missed floods cost lives and property. The balance is a policy decision, not only a statistical one.'));
kids.push(B('No personal data was used. The model is a teaching prototype and must not replace official warnings from the Disaster Management Centre or the Irrigation Department.'));
kids.push(H2('5.4 Potential real-world impact'));
kids.push(P('As a screening layer on top of existing forecasts, a model like this could help prioritise which rivers to watch closely. For that, the next steps would be to validate against gauge data, add rainfall, extend the record to 10 or more years, evaluate by basin and calibrate the probabilities.'));

kids.push(H1('6. Front-end integration (bonus)'));
kids.push(P('A Streamlit application (app/app.py) wraps the selected LightGBM model. It has three tabs: (1) choose a river point and a date and see the risk score, the alert decision, the actual outcome in the data and a SHAP waterfall explanation; (2) enter recent flows and the high-flow level by hand and get a risk score with its explanation; (3) the model comparison table and figures. The app was tested automatically (no exceptions; changing the inputs changes the score). Run it with streamlit run app/app.py.'));
kids.push(H1('References'));
for(const t of ['Copernicus Emergency Management Service. GloFAS v4 historical reanalysis (river discharge), Copernicus Early Warning Data Store, https://ewds.climate.copernicus.eu.','Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30.','Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30.','Nori, H., Jenkins, S., Koch, P. and Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. arXiv:1909.09223.','Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.'])kids.push(P(t));
kids.push(H1('Appendix: reproducibility'));
kids.push(P('All code is in the submitted ZIP: src/prepare.py builds the table, src/train.py trains and compares the models (random seed 42), src/explain.py produces the SHAP and partial dependence figures, and app/app.py is the front-end. requirements.txt lists the package versions used.'));
const doc=new Document({creator:'',title:'Flood-onset early warning',styles:{default:{document:{run:{font:'Calibri',size:22}}},paragraphStyles:[
 {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:32,bold:true,color:'1F3864'},paragraph:{spacing:{before:320,after:160},outlineLevel:0,keepNext:true}},
 {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:26,bold:true,color:'2F5496'},paragraph:{spacing:{before:220,after:100},outlineLevel:1,keepNext:true}}]},
 numbering:{config:[{reference:'bul',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:600,hanging:300}}}}]}]},
 sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1134,bottom:1134,left:1134,right:1134}}},footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({children:['Page ',PageNumber.CURRENT],size:18,color:'777777'})]})]})},children:kids}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(ROOT+'/Flood_Onset_ML_Assignment_Report.docx',b);console.log('written');});
