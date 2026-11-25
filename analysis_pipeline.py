from pathlib import Path
import shutil, numpy as np, pandas as pd, pyreadr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; FIG=ROOT/'figures'; OUT=ROOT/'outputs'
for p in [DATA,FIG,OUT]: p.mkdir(exist_ok=True)
SOURCE=Path('/tmp/completejourney/data')
for n in ['transactions.rds','products.rda','demographics.rda']: shutil.copy2(SOURCE/n,DATA/n)
tx=pyreadr.read_r(str(DATA/'transactions.rds'))[None]; products=pyreadr.read_r(str(DATA/'products.rda'))['products']; demo=pyreadr.read_r(str(DATA/'demographics.rda'))['demographics']
tx['transaction_timestamp']=pd.to_datetime(tx['transaction_timestamp']); cut=pd.Timestamp('2017-10-01')
feature=tx[tx.transaction_timestamp<cut].copy(); holdout=tx[tx.transaction_timestamp>=cut].copy()
g=feature.groupby('household_id')
f=g.agg(last_purchase=('transaction_timestamp','max'),frequency=('basket_id','nunique'),monetary=('sales_value','sum'),items=('quantity','sum'),distinct_products=('product_id','nunique')).reset_index()
f['avg_basket']=f.monetary/f.frequency; f['recency']=(cut-f.last_purchase).dt.days
cat=feature.merge(products[['product_id','department','product_category']],on='product_id',how='left')
c=cat.groupby('household_id').agg(category_count=('department','nunique'),product_category_count=('product_category','nunique')).reset_index()
f=f.merge(c,on='household_id',how='left'); ret=holdout.groupby('household_id').size().clip(upper=1).rename('retained').reset_index()
cust=demo.merge(f,on='household_id',how='left').merge(ret,on='household_id',how='left'); cust['retained']=cust.retained.fillna(0).astype(int)
cust['family_size']=pd.to_numeric(cust.household_size,errors='coerce'); num=['recency','frequency','monetary','avg_basket','items','distinct_products','category_count','product_category_count','family_size']; cust[num]=cust[num].fillna(cust[num].median())
sc=StandardScaler(); Xs=sc.fit_transform(cust[num]); pca=PCA().fit(Xs); scores=pca.transform(Xs)
load=pd.DataFrame(pca.components_.T,index=num,columns=[f'PC{i+1}' for i in range(len(num))]); var=pd.DataFrame({'component':[f'PC{i+1}' for i in range(len(num))],'explained_variance':pca.explained_variance_ratio_,'cumulative_variance':np.cumsum(pca.explained_variance_ratio_)}); load.to_csv(OUT/'pca_loadings.csv'); var.to_csv(OUT/'pca_variance.csv',index=False)
diag=[]
for k in range(2,7):
    m=KMeans(n_clusters=k,n_init=30,random_state=42).fit(scores[:,:5]); reps=[KMeans(n_clusters=k,n_init=20,random_state=s).fit(scores[:,:5]).labels_ for s in [1,2,3,4,5]]; stability=np.mean([adjusted_rand_score(m.labels_,z) for z in reps]); diag.append({'k':k,'inertia':m.inertia_,'silhouette':silhouette_score(scores[:,:5],m.labels_),'min_cluster':np.bincount(m.labels_).min(),'stability_ari':stability})
kd=pd.DataFrame(diag); kd.to_csv(OUT/'k_diagnostics.csv',index=False); best_k=4  # k=2 has the best silhouette, but k=4 is more actionable for the required strategy framework.
km=KMeans(n_clusters=best_k,n_init=50,random_state=42).fit(scores[:,:5]); cust['cluster']=km.labels_+1
for i in range(5): cust[f'PC{i+1}']=scores[:,i]
prof=cust.groupby('cluster').agg(customers=('household_id','size'),recency=('recency','median'),frequency=('frequency','median'),monetary=('monetary','median'),avg_basket=('avg_basket','median'),return_rate=('retained','mean')).reset_index(); prof['pct_customers']=prof.customers/len(cust)
hv=int(prof.loc[prof.monetary.idxmax(),'cluster']); risk=int(prof.sort_values('recency',ascending=False).cluster.iloc[0]); low=int(prof.sort_values('monetary').cluster.iloc[0]); labels={hv:'High-Value Loyal Customers',risk:'Recently Lapsed / At-Risk Customers',low:'Low-Value Low-Touch Customers'}
for cl in prof.cluster:
    if cl not in labels: labels[cl]='Middle-Value Growth Customers'
cust['segment']=cust.cluster.map(labels); prof['segment']=prof.cluster.map(labels); prof.to_csv(OUT/'segmentation_profile.csv',index=False)
Xtr,Xte,ytr,yte=train_test_split(cust[num],cust.retained,test_size=.25,random_state=42,stratify=cust.retained); model=LogisticRegression(max_iter=2000).fit(Xtr,ytr); pr=model.predict_proba(Xte)[:,1]; pred=(pr>=.5).astype(int); cm=confusion_matrix(yte,pred); auc=roc_auc_score(yte,pr)
metrics=pd.DataFrame([{'model':'Original behavior variables','AUC':auc,'sensitivity':recall_score(yte,pred),'specificity':cm[0,0]/cm[0].sum(),'precision':precision_score(yte,pred,zero_division=0),'cutoff':.5}]); metrics.to_csv(OUT/'model_metrics.csv',index=False); pd.DataFrame(cm,index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1']).to_csv(OUT/'confusion_matrix.csv')
cust['predicted_return_probability']=model.predict_proba(cust[num])[:,1]
alloc={'High-Value Loyal Customers':250000,'Middle-Value Growth Customers':400000,'Recently Lapsed / At-Risk Customers':250000,'Low-Value Low-Touch Customers':100000}; actions={'High-Value Loyal Customers':'Recognition, early access, personalized cross-sell','Middle-Value Growth Customers':'A/B-test bundles, cross-sell, and moderate incentives','Recently Lapsed / At-Risk Customers':'Low-cost win-back test with response measurement','Low-Value Low-Touch Customers':'Automated reminders and low-cost seasonal communication'}
seg=cust.groupby('segment').agg(customers=('household_id','size'),predicted_return_rate=('predicted_return_probability','mean'),value_measure=('monetary','median')).reset_index(); seg['marketing_action']=seg.segment.map(actions); seg['allocation']=seg.segment.map(alloc); seg['percent_budget']=seg.allocation/1000000; seg['justification']='Balances value protection, churn risk, growth potential, and contact cost.'; seg.to_csv(OUT/'budget_allocation.csv',index=False)
cust.to_csv(OUT/'customer_level_features.csv',index=False)
sns.set_theme(style='whitegrid', palette='Set2'); plt.figure(figsize=(7,4)); plt.fill_between(var.component,var.cumulative_variance,color='#8ecae6',alpha=.35); plt.plot(var.component,var.cumulative_variance,marker='o',color='#126782',linewidth=2.5); plt.axhline(.8,color='#f4a261',ls='--',linewidth=2); plt.title('PCA cumulative explained variance'); plt.tight_layout(); plt.savefig(FIG/'pca_variance.png',dpi=160); plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot(kd.k,kd.inertia,marker='o',color='#7b2cbf',linewidth=2.5); ax[0].set_title('K-means elbow'); ax[1].plot(kd.k,kd.silhouette,marker='o',color='#e76f51',linewidth=2.5); ax[1].axvline(best_k,color='#2a9d8f',ls='--',linewidth=2); ax[1].set_title('Silhouette by k'); plt.tight_layout(); plt.savefig(FIG/'k_diagnostics.png',dpi=160); plt.close()
palette={'High-Value Loyal Customers':'#264653','Middle-Value Growth Customers':'#2a9d8f','Recently Lapsed / At-Risk Customers':'#e76f51','Low-Value Low-Touch Customers':'#e9c46a'}; plt.figure(figsize=(8,5)); sns.scatterplot(data=cust,x='PC1',y='PC2',hue='segment',palette=palette,size='monetary',sizes=(20,180),alpha=.75); plt.title('Customer segments in PCA space'); plt.tight_layout(); plt.savefig(FIG/'cluster_map.png',dpi=160); plt.close()
fpr,tpr,_=roc_curve(yte,pr); plt.figure(figsize=(6,5)); plt.plot(fpr,tpr,label=f'AUC={auc:.3f}',color='#264653',linewidth=2.5); plt.plot([0,1],[0,1],'--',color='#e9c46a'); plt.legend(); plt.title('Logistic regression ROC curve'); plt.tight_layout(); plt.savefig(FIG/'roc_curve.png',dpi=160); plt.close()
pt,pp=calibration_curve(yte,pr,n_bins=5); plt.figure(figsize=(6,5)); plt.plot(pp,pt,marker='o',color='#e76f51',linewidth=2.5); plt.plot([0,1],[0,1],'--',color='#2a9d8f'); plt.xlabel('Mean predicted probability'); plt.ylabel('Observed return rate'); plt.title('Calibration plot'); plt.tight_layout(); plt.savefig(FIG/'calibration.png',dpi=160); plt.close()
print({'customers':len(cust),'feature_transactions':len(feature),'holdout_transactions':len(holdout),'best_k':best_k,'auc':round(auc,3),'labels':labels})
