- Based on the Samuel, 1959, Machine Learning algorithm enable the computers to **learn** from data, and even **improve**  themselves, without being explicitly **programmed**. 
  
- Tom Mitchell, 1977: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. 
---
Two approach can be use two solve a problem in programming.
 
- Traditional way: 
 
- Machine Learning :
---
We  can use ML where:
1. Finding the rule of the problem is difficult 
2. No other method can solve the problem
3. High oscillating problem 
4. Situations with a Large amount of data
---
### Variations of ML 
1. **Supervised Learning**: Data has it's own solution of label or target.
2. **Unsupervised**: Data don't have any label. We can compress data or some variations on the dimensions of the data, Clustering... . 
   To say it n the simple way, you don't Know what you want.
3. **Reinforcement Learning**: Machine Learns based on the choose and result of the agent.
---
### Systems of ML
1. Instance-Based: Model compares each data point with other data points to find a model.
2. Model-Based: Model of Machine use a basic and fundamental model(e.g. a physics model) to find the rule of learning.  
---
### Challenges of  ML
1. Lack of data 
2. Non-representative Data: your data doesn't represent the model
3. Low quality Data(Noisiness)
4. Featureless Data
5. Over-fitting: TOO complex model for data
6. Under-fitting: TOO simple model for the data
---
### Validation and Test of the data 
#### Hyperparameters configuration
We need to separate the data into 3 part;
1. Train: Input data for training 
2. Validation: input data for prediction of output result.
3. Test: This part wont go to the data. so after the training we show it to the model and find an output result to ignore bias.
---
### Regression 
Since a major category  of machine learning is supervised learning with a known target we can categorized it into
1. **Regression**: for Continuous variables.
2. **Classification**: for discrete variables.
--- 

#### Steps of ML
1. DATA: 
   I. Data gathering
   II. Data analysis --> Pandas 
   III. Data cleaning 
   IV. make ready for using data in tho model
2. Model:
---
### Data for Machine Learning
1. Data gathering:  
2. Data Analysis: Visualization and finding some parameters
3. Make data ready for ML algorithm:
- Data cleaning
- text data 
- scaling the data: like normalizing the data 
---
### Resources for Data 
- kaggle.com
- openml.org
- paperswithcode.com
- amazon aws dataset 
- tensorflow.org 
---
### Step by Step on data of California home market
```python
# Import libraries
import numpy as np
import pandas as pd

# Read the csv file using pandas 
pd.read_csv("directory of the csv data")
# You can put this data that pandas has been read into a vairiable df(data frame)
df = pd.read_csv("/home/directories/...")
```
You can see the information of the data using `.info()`:
```python
df.info()
```
After that you can even call the row or column of the data by using `df.<nameofthetitle>`
```python
df.ocean_proximity.value_counts()
# This will show the number of values of title ocear_proximity
```

You can even describe some information of the data set using `.describe`in pandas
```python
df.describe()
```
This will show you some statistical properties of your data set for example meadian, max, min, ... .

---
### Visualization
Visualizing the data can be helpful to understand the problem and finding the solution. 
Most famous library for visualizing the data is `matplotlib` with multiple modules and properties. for example for the dataset df above we can use `.hist` to plot a histogram of the df:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("directory of the dataset")

# Plotting the histogram of your data frame
df.hist(bins=40, figsize=(11,7))  
```

### Dividing data into train and test set
#### Test set
We can(should) divide data to test set to see the result of the model in our data with out model see the complete data. we should do it **randomly**.
```python
random_indices = np.random.permutation(len(df))
# This will give us data randomly in our data set
test_set_size = int(len(df)*0.2) 
test_random_indices = random_indices[:test_set_size]
# SO the train set will become:
train_random_indices = random_indices[test_set_size:]
# These are just indix, we need data itself
df.iloc[train_random_indices]
# OR
df.iloc[test_random_indices]
```
Let's write a function for above, a ready to use one:
```python
def shuffle_and_split_df(df, test_ratio):
	random_indices = np.random.permutation(len(df))
	test_set_size = int(len(df)*test_ration) 
	test_random_indices = random_indices[:test_set_size]
	train_random_indices = random_indices[test_set_size:]
	return df.iloc[train_random_indices], df.iloc[test_random_indices]
```
for finding a constant and static result we can set a constant random seed of save split data.
```python
def shuffel_and_split_df(df, test_ration):
# We set a constant seed to get static result
	np.random.seed(40)
	test_set_size = int(len(df)*test_ration) 
	test_random_indices = random_indices[:test_set_size]
	train_random_indices = random_indices[test_set_size:]
	return df.iloc[train_random_indices], df.iloc[test_random_indices]

```
But we want to split completely test and train data set. if we change the data set, sometimes train data move to test data. What should we do to solve this problem?

We can use **zlib library and module crc32**.
In this case we can track the data blocks by giving id(Hash) to data. here we want to give a hash(identifier) to each line so we can track each line by their id.
```python
from zlib import crc32
# crc32 turn every thing to 32 bites 
def is_identifier_in_test_set(identifier, test_ration):
	return crc32(np.int64(identifier)) < test_ration*2**32
```
now we can split data to test and train dataset even for updating data:
```python
def split_train_test_with_identifier_hash(df, test_ratio, identifier_column):
	identifiers = df[identifer_column]
	in_test_set = identifier.apply(lamba id_: is_identifier_in_test_set(id_, test_ration))
	return df.loc[in_test_set], df.loc[in_test_set]
```
for train dataset we can simply use **not** in above function:
```python
def split_train_test_with_identifier_hash(df, test_ratio, identifier_column):
	identifiers = df[identifer_column]
	in_test_set = identifier.apply(lambda id_: is_identifier_in_test_set(id_, test_ration))
	return df.loc[~in_test_set], df.loc[in_test_set]
```
خب حالا بحث اینه که ما اینجا ستون آیدنتیفایر نداریم. میتونیم براش یه ستون درست کنیم که اون ستون آیدنتیفایر ها رو نشون بده بهم. برای مثال میتونیم:
```python
df.reset_index()
```
این میاد اندیس ها رو هم به دیتا فریممون اضافه میکنه. برای مثال با استفاده از این ایندکس گذاری بالا میتونیم اسپلیت کردن دیتای تست و ترین رو به صورت پایین بنویسیم:

```python
split_train_test_with_identifier_hash(df.reset._index(), 0.2, "index")
```
حالا یه مشکلی که استفاده از اندیس سطرها به عنوان آیدنتیفایر داره اینه که در صورتی کار میکنه که همیشه دیتای جدید رو به انتهای دیتای قبلی اضافه کنیم و در غیر این صورت همه چیز به هم میریزه و جای اندیس ها عوض میشه. باید چیکار کنیم؟ کار منطقی اینه که خودمون با استفاده از یه ترکیبی از خصوصیات دیتامون که برای هر سطر از دیتا اختصاصی هست یه آیدنتیافیر درست کنیم. برای مثال توی دیتای خونه ها میتونیم از ترکیب طول و عرض جغرافیایی استفاده کنیم چرا که طول و عرض تغییر نمیکنن و برای هر سطر دیتا منحصر به فرد هست. برای مثال:
```python
df_with_identifier = df
df_with_identifier["identifier"] = df["longitude"]*1000 + df["latitude"]
```
اینجوری عملا یه ستون آیدنتیفایر اختصاصی خودمون رو درست میکنیم. حالا بر اساس این آیدنتیفایر جدید میتونیم ترین و تست دیتا رو اسپلیت کنیم.
```python
train_set, test_set = split_train_test_with_identifier_hash(df_with_identifier, 0.2, "identifier")
```
حالا دیگه با آپدیت کردن  دیتا فریم و تغییر دیتا فریم به مشکلات ذکر شده  نمیخوریم و همیشه با هش درست شده میتونیم بدون مشکل دیتا فریممون رو به تست و ترین  اسپلیت کنیم.  
آیا این همه کاره؟ آسون ترین کاره؟ نه. یه کتاب خونه خیلی مهم توی ماشین لرنینگ هست به اسم sikit-learn یا sklearn که خودش در یه ماژول این اسپلیت کردن رو داره که میتونیم به صورت ساده ازش استفاده کنیم به صورت:
```python
from sklearn.model_selection import train_test_split(df, test_size = 0.2 , random_state = 40)
```
که رندوم استیت عملا همون سید نامبر هست. 
### Stratified Sampling
نکته ای که برای پروسه ای که نوشتیم وجود داره اینه که بالاخره توی هر کدوم از این ها برای دیتاست های بزرگ کار کنیم این کارایی که کردیم خوبه اما اگر دیتاستمون خیلی بزرگ نبود باید چیکار کنیم؟ به خصوص اگر دیتا ستمون نسبت به فیچر ها کم باشه چی؟ این موقع پدیده sampling bias که توی آمار هست به وجود میاد.
#### Sampling bias
فرض کنیم تو یه شرکت میخوایم با هزار نفر از مشتری ها میخوایم تماس بگیریم. نمیایم این هزار نفر رو کاملا رندوم انتخاب کنیم. ممکنه مثلا ۵۲ درصد از مشتری ها خانوم باشه وو ۴۸ درصد خانوم. در این صورت ۵۲۰ نفر باید خانوم باشه و ۴۸۰ نفر آقا باشن. اگر اینکارو نکنیم و رندوم انتخاب کنیم سمپلینگ بایاس به وجود میاد. 
بنابر این راهکار اینه که نسبت درصد یه اتریبیوت توی دیتای train با درصد اون اتریبیون توی دیتای تست هم یکی باشه.
این کاری هست که ما بهش میگیم stratified sampling یا نمونه برداری طبقه بندی شده که توش کاملا رندوم سمپلینگ نمیکنیم و دسته بندی ها رو هم توی سمپلینگ در نظر میگیریم. فرض کنید یه فیچر توی دیتامون خیلی اهمیت داره و باید بر اساس اون استرتیفاید سمپلینگ انجام بدیم. 
توی دیتاستی که داریم مثلا median_income  میتونه برامون مهم باشه که یه فیچر پیوسته هم هست.
اگه داده اون فیچر ما پیوسته باشه میتونیم بین بندی کنیم (همون کاری که تو کیهان شناسی من انجام میدم).

میشه خیلی ساده از سایکیت لرن استفاده کرد ولی با پانداس هم میشه:
```python
pd.cut(df["median_income"], bins = [0.0,1.5,3.0,4.5,6.0, np,inf],
labels = [1,2,3,4,5])
```
اینجوری خروجی توی هر بین یه لیبل هم میگیره. عملا ما اینجوری میتونیم سمپلینگ دیتا رو وزندار(weighted) کنیم.  
### Stratified Sampling implementation
برای اینکار میتونیم از سایکیت لرن اضافه کنیم. 
```python
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_split = 10, test_size = 0.2, random_state = 40)
stratified_split = []
for train_im test_i in sss.split(df, df["median_income_categories"]):
	stratified_train_set_n = df.iloc[train_i]
	stratified_test_set_n = df.iloc[test_i]
	stratified_split.append([stratified_train_set_n, stratified_test_set_n])
```
این چی بر میگردونه؟ یه اسپلیتر برمیگردونه که متد اسپلیت داره و متد اسپلیت به ما اندیس میده. بعد  از اون باید بیایم این کتگوری هایی که درست کردیم رو حذف کنیم چون دیگه نیاز نداریم و فقط برای سمپلینگ استفاده شده. یه راهی که داریم استفاده از drop هست:
```python
str_train_set = str_train_set.drop("median_income_categories", axis = 1)
str_test_set = str_test_set.drop("median_income_categories", axis = 1)
```
### Visualization
حالا که داده رو بهتر شناختیم حالا میخوایم یکم دقیق تر داده رو بررسی کنیم و مصور سازی بهتری کنیم تا مشخص تر ببینیم چی تو داده ما هست.
اولا که میتونیم با توجه به وجود دیتای طول و عرض جغرافیایی بیایم اینها رو با یه اسکتر پلات رسم کنیم. یکی از راه هاش استفاده از seaborn هست.
```python
train = str_train_set
sns.scatterplot(data = train, x="longitude" , y = "latitude")

```
کاری که میتونیم انجام بدیم اینه که نوع پلات کردن رو یکم تنظیم کنیم. مثلا میتونیم توی پلات کردن رنگ رو مربوط بدیم به پاپیولیشن و جمعیت. چیکار کنیم؟‌
```python
train = str_train_set
ax = sns.scatterplot(data = train, x="longitude" , y = "latitude", size = "population", alpha = 0.2, hue="median_house_value")
sns.move_legend = (ax, "upper left", bbox_to_anchor = (1,1))
#نکته اینکه چون این اطلاعات تو تصویر بود و تصویر رو ناخوانا میکرد ما با استفاده از ابزار موولجند سیبورن اونو گذاشتیم کنار تصویر که هم تصویر خوانا باشه هم لجندش.
```
!!!!یه آموزش از سیبورن اضافه بشه اینجا به طور خلاصه
### Correlation
ممکنه پارامتر ها با هم ارتباط داشته باشن بنابراین ممکنه که نیاز باشه کورولیشن رو بین پارامتر ها حساب کنیم. در واقع کورولیشن نحوه ارتباط بین دوتا متغیر رو بهمون بده. که به صورت:
$$
\rho_{X,Y} = \mathrm{corr}(X,Y)
= \frac{\mathrm{cov}(X,Y)}{\sigma_X \sigma_Y}
= \frac{\mathbb{E}\!\left[(X-\mu_X)(Y-\mu_Y)\right]}{\sigma_X \sigma_Y},
\qquad \text{if } \sigma_X \sigma_Y > 0 .
$$

![[Pasted image 20260426114904.png]]
نکته ای که هست اینه که کورولیشن یک رابطه خطی رو به ما نشون میده و نوع استقلال بین پارامتر ها رو به طور دقیق به ما نمیده. توی پانداس باید چیکار کنیم؟
```python
train.corr()
```
که این دستور میاد کورولیشن بین همه پارامتتر ها رو جدا جدا بهمون میده. درواقع به شکل یک شبه  ماتریس بهمون میده این کورولیشن ها رو. 
چجوری باید کورولیشن رو تفسیر کنیم؟ چیزی که تو آمار خطرناک هست اینه که تفسیر مستقیم داشته باشیم بابت کازالیتی برای مثال تفسیر کنیم اگر دوتا پارامتر کورولیشن زیاد هست بگیم یکی باعث ایجاد اون یکی هست. این نوع تفسیر اشتباه هست و نمیشه اینجوری استنتاج علی انجام بدیم. 
ارتباطات علت و معلولی کار بسیار سخت تری هست.  
چجوری کورولیشن ها رو رسم کنیم؟
```python
sns.scatterplot(data = train, x="median_income", y="median_house_value" , alpha = 0.2)

```
برای  خط های افقی بالای نمودار که ممکنه به دست بیاد در کل نمودار مثلا ممکنه علتش این باشه که اومدن قیمت ها رو رند کردن و نوشتن و باعث این خطوط شده.
![[Pasted image 20260426115843.png]]
کاری که میتونیم بکنیم اینه که این خطوط رو حذف کنیم چون بایاس به وجود میاره توی پیشبینی مدل

### Attribute Combination
گاهی فیچرهامون مثلا تعداد کلی اتاق ها و تعداد کلی خانوار ها رو داریم ولی این شاید مناسب نباشه و مناسب ترش این باشه که بیایم تعداد اتاق ها رو برای هر خونه با توجه به تعداد خانوارها  به دست بیاریم.
```python
train["rooms_per_house"] = train["total_rooms"] / train["total_households"]
train["bedrooms_ratio"] = train["total_bedrooms"]/ train["total_rooms"]
train["peaple_per_house"] = train["population"] / train["households"]
```
حالا که دیتا رو به طور کلی بررسی کردیم حالا میتونیم بریم سراغ آماده سازی دیتامون برای وارد کردنش به مدل.
### Deleting Missing values 
#### prepare data
چند مرحله داره که قدم به قدم باید بریم جلو و از امکانات تنسور فلو استفاده میکنیم.
قدم اول بیایم یک سری از ستون هایی که لازم نداریم رو حذف کنیم. مثلا:
```python
train_features = train.drop("median_house_value", axis =1 )
train_target = train["median_house_value"]
```
تا الان ورودی مدل و چیزی که باید خروجی بده رو جدا کردیم. 
حالا باید یکم دیتا رو تمیز کنیم:
#### Data cleaning 
چیزی که منظور هست اینه که ببینیم کجا میسینگ ولیو داریم. یعنی جاهایی که NA یا NAN داریم و جاهایی رو که خالی هستن مشخص کنیم. 
```python
train_features.info()
```
با توجه به خروجی این باید مثلا  اون ستون  هایی که مقادیر ما رو نداره  حذف کنیم. ممکنه این خوب نباشه لزوما چون منجر به از دست رفتن اطلاعات یک فیچری که میخوایم بشه. میشه یک درصدی رو حذف کرد مثلا ۵ درصد. ولی خب برای دقیق شدن میشه سطرهای خالی رو فقط حذف کرد. 
کار دیگه ای که میشه اینه که اونجاهایی که مقدار نیست رو خودمون بیایم مقدار بذاریم. مثلا بیایم میانگین رو بذاریم. 
بیایم اول اونایی که توی سطر ها مقدار خالی هست رو حذف کنیم.

```python
#remove rooms with NA values with dropna
train_feature.dropna(subset=["total_bedrooms"]).info()
```
حالا بیایم جای مسیسنگ ولیو ها یه مقدار منطقی بذاریم. مثلا بیایم مدین اون ستون رو بذاریم.
```python
total_bedrooms_median = train_features["total_bedrooms"].median()
# train_features_with_imputed_na_value
train_features_with_imputed_na_value = total_features["total_bedrooms"].fillna(total_bedrooms_median)
```
یه امکانی از سایکیت لرن که کار ما رو راجت تر میکنه امکان imputer هست. 
#### sklearn impute: sklearn.impute.Simpleimputer
تمرین
```python
from sklearn.impute import SimpleImputer
simple_imputer = SimpleImputer(strategy = "median")
```
نکته: اگه ابزار پی پراسس استفاده میکنیم بهتره اول یه آبجکت درست کنیم به اسم اون ابزار و بعد دیتاها رو بریزیم تو اون. مثل مثال بالا. جالا میخوایم همینکارو روی دیتای ترین فیچر ها انجام بدیم. 
```python
train_features_numeric = train_features_dtype(include = [np.number])
#با اینکار بالا فقط دیتاهای عددی رو نگه میداریم چون مدل فقط دیتا میبینه.
simple_imputer.fit(train_features_numeric)
# میاد نتایج رو فیت میکنه رو این آبجکتی که درست کردیم برای ابزارمون
simple_imputer.transform(train_features_numeric)
pd.DataFrame(simple_imputer.transform(train_features_numeric), columns=train_features_numeric.columns,
index = train_features_numeric.index)
```

### Other methods of Impute
میشه به جای صرفا سیمپل از روش های دیگه ای هم توی متد ایمپیوت استفاده کرد مثلا میشه از خود مدل استفاده کرد برای پر کردن و یا مدل مشخص مثلا knn. مدل های مختلفی هست که تو داکیومنتیشن ها موجوده.

### Sklearn architucure
یکی از ویژگی های خوب سایکیت لرن طراحی خیلی خوبش هست. یکی از این ویژگی ها:
#### Consistency
  یک اینترفیس ساده و شبیه به هم استفاده میکنن. مثلا:
  - estimator:
هر چیزی که بتونه بر اساس آبجکت یه پارامتر رو تخمین بزنه یه استیمیتور هست.
هر کدوم از این استیمیتور ها یه متد ‍`fit‍` دارند.
- transformer:
بعضی از این استیمتورها میتونه ترنسفورمر باشن یعنی یه دیتا رو بگیره وتخمین بزنه و بعد از اون تخمین استفاده کنه برای استفاده دوباره از دیتا. به نوعی دیتا رو ترنسفورم میکنه و بهم یه دیتاست جدید میده بهمون.
- Predictor:
مدل هایی که داریم مثل knn یه متد فیت دارن یعنی مدل رو روی دیتا فیت میکنن و بعد پارامتر رو پیشبینی میکنن.
#### Inspection
تمامی ویژگی هایی که توی سایکیت لرن استفاده میکنیم یه سری هایپرپارامتر دارن که میتونیم اونا رو به طور دقیق تنظیم کنیم. 

سوالی که ما رو به مبحث بعدی میبره اینه که با داده های کتگوریکال که عدد نیستن توی ماشین لرنینگ باید چیکار  کنیم؟ چون ماشین صرفا عدد میفهمه.

### Ordinal Encoder
همونطور که گفته شد مدل ها صرفا عدد رو میفهمن. باید یک سری دیتاها که عدد ندارد و کتگوریکال هستن رو به نحوی تبدیل به عدد کنیم.
برگردیم به دیتامون:
```python
# categorical features
train_features[["ocean_proximity"]]
```
که خروجی به صورت رشته هستند. حالا بیایم یه اسم جدید بهشون بدیم. کار زیر رو معمولا انجام میدیم که توی دیتا مشخص کنیم با چی طرف هستیم و عدد ها و رشته ها رو به نوعی از هم جدا کنیم چون که قراره کارهای مختلفی روشون انجام بشه. پس حالا یه دیتافریم کتگوریکال داریم.
```python
train_features_categorical = train_features[["ocean_proximity"]]
```
حالا میتونیم برای هر کدوم از این مقادیر یه عدد بذاریم. مثلا اگه جزیره بود بذاریم صفر و برای بقیه هم همینطور. عملا داریم به هر ویژگی رشته ای یه عدد ایندکسی مشخص کنیم. این متد برای وقتی که ۵ تا کتگوری داریم ساده هست ولی اگه ۵۰ تا باشه چی؟ از امکانات سایکیت لرن استفاده میکنیم. یکی از ابزارها:
```python
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
ocean_proximity_index = oe.fit_transform(train_features_categorical)
oe.categories_
```
یه مشکلی که هست اینه اگر بین کتگوری ها ترتیبی وجود داشته باشه مشابه به بد، خوب، عالی مدل میتونه خوب بیاد اینا رو تبدیل به عدد کنه.
ولی وقتی مثل بالا این ترتیب ملموس نباشه باید کار دیگه ای کرد:
```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ocean_proximity_ohe = ohe.fit_transform(train_features_categorical)
```

حالا بیایم این دیتای جدید رو به دیتافریممون اضافه کنیم:
```python 
train_features_categorical_ohe = pd.DataFrame(ocean_proximity_ohe.toarray(),columns = ohe.get_features_names_out())
```
نکته اینکه در بحث فیت ترنسفورم همیشه روی داده ترین اعمالش میکنیم یعنی روی ترین فیت میکنیم و بعد روی داده تست ترنسفورم میکنیم. علتش هم اینه که میخوایم داده تست کاملا برای مدلمون ناشناخته باشه. 

### OneHotEncoder

مشکلی که autoencoder داره اینه اختلاف بین اعداد برای مدل اهمیت داره و از روی این اختلافات قرار هست قوانین خودش رو بسازه. 
حالا وقتی به شکل autoencoder کتگوری ها رو تبدیل به عدد میکنه ممکنه مدل بگه چون اختلاف عددی  دوتا کتگوری بیشتره پس اون دوتا کتگوری خیلی با هم متفاوت هستن از نظر معنایی. تو یه حالت هایی این کاملا درستته و اتفاقا ما همینو میخوایم مثلا یه جایی کتگوری هایی داریم مثل سن یا مثلا کودک تا بزرگسال. تو این حالت واقعا نوجوان و کودک از نظر عددی نزدیک هستن نسبت به کودک و بزرگسال. پس یک جاهایی این واقعا درسته و مورد نیاز ما هست و autoencoder کار میکنه.
اما!
یک جاهایی کتگوری های ما اون معنای خاص مثل بالا رو ندارن که بر اساس فاصله عددی بین اون دوتا کتگوری خیلی اهمیت پیدا کنه مثل همین داده هایی که داریم برای نزدیک به اقیانوس داشتیم. مثلا کتگوریمون وقتی کشورها باشه لزوما اوردر خاصی نداریم براش. 
باید دید که ترتیبی آیا در کتگوری ها هست؟ اگر آره اتوانکودر خوبه. اگر نیست باید بریم از OneHotEncoder رو استفاده کنیم.

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHatEncoder()
ocean_proximity_ohe = ohe.fit_transform(train_features_categorical)
```
خروجی این یک اسپارس ماتریکس هست ولی ممکنه ما بخوایم آرایه‌اش رو به دست بیاریم. برای اینکار از `toarray()` استفاده میکنیم.
```python 
ocean_proximity_ohe.toarray()
```
که به ما یه آرایه دو بعدی برمیگردونه.

یه نکته ای که هست اینه که ما هیچ وقت `fit.transform` رو روی داده تست نمیزنیم بلکه روی train فیت میکنیم و بعد روی داده test ترنسفورم میکنیم.
### MinMax Scalar 

یکی از مهم ترین ترنسفورمیشن هایی که باید انجام بدیم اصطلاحا ‍`feature scaling` هست. تقریبا اغلب الکورتیم های یادگیری ماشین وقتی دیتاست هایی  داریم که اسکیل های خیلی مختلفی دارن خیلی خوب کار نمیکنند. برای مثال توی داده های ما `househol` مثلا اسکیلش ۲۰۰۰ تایی هست ولی یهوی عرض جغرافیایی ۳۰و۳۵ تایی هست. خیلی اسکیل هاشون متفاوته در فیچر های مختلف داده‌امون. 
خب حالا برای اینکه این فیچرهای مختلف با اسکیل های متفاوت ارجحیت خاص تری پیدا نکنن باید چیکار کنیم؟ ساده است. بیایم همشون رو اسکیل کنیم به اندازه مقدار مشخصی مثلا همشون رو بیایم اسکیل کنیم بین صفر و یک. 
به اینکار میگن `Normalization` که با در سایکیت لرن با ابزار `MinMax Scaler` انجام میشه.

```python
# MinMax Scaler

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(-1, 1))
mms.fit_transform(train_features_numeric)
```

که خروجی زیر را دارد:
```txt
array([[ 0.24701195, -0.7130712 , 0.60784314, ..., -0.93355263, -0.5603371 , 0.24647561], [ 0.25896414, -0.74070138, -0.05882353, ..., -0.74078947, -0.2769617 , 0.25840945], [-0.50199203, -0.0053135 , -0.21568627, ..., -0.7375 , -0.00412408, -0.50233863], ..., [-0.5498008 , 0.09883103, 0.33333333, ..., -0.87993421, -0.35864333, -0.55007998], [-0.00398406, -0.19659936, -0.64705882, ..., -0.76578947, -0.54157874, -0.00419488], [-0.51992032, -0.05419766, -0.60784314, ..., -0.72203947, -0.47726238, -0.52032411]], shape=(16512, 9))
```

که اگر دقت شود همه آنها بین منهای یک تا یک هستند.

### Standardization (standard scaler)

از مشکلات `MinMaxscaler` وابسته بودن زیاد به outlaier هاست یعنی فرض کنید اگه یک فیچر داشته باشیم که مثلا از یک باشه یکیشون تا ۱۰۰ اما یک سری از مواردش خیلی از این بازه بیرون هستند مثلا یهویی یکیش ۵۰۰ هست. یکیش ۲۰۰۰ هست. اینا خیلی پرت هستن. خب مشکل چی میشه؟ 
تو این حالت این MinMax میاد اون ۵۰۰ رو میذاره یک و بقیه رو نسبت به اونا تغییر میده. ولی خب اینجوری بین ۱۰۰ تا ۵۰۰ عدد زیادی نیست و این بازه خالی میمونه پس این روش قابل اتکا نیست لزوما. خب باید چیکار کنیم؟ 

یکی از راه هاش استفاده از `standard scaler` هست که از ابزار های سایکیت لرن هست.
Standardize features by removing the mean and scaling to unit variance.
The standard score of a sample `x` is calculated as:
z = (x - u) / s
where `u` is the mean of the training samples or zero if `with_mean=False`, and `s` is the standard deviation of the training samples or one if `with_std=False`.‍‍‍
پس عملا میایم توزیع داده رو میاریم روی توزیع گاوسی یا همون استاندارد و دیگه حساسیت قبل رو نداره هر داده رو از مقدار میانگین کم میکنیم و از استاندارد دیویشن تقسیم میکنیم. اینجوری این حساسیت کم میشه چون بیشتر مقادیر دیگه رفته اطراف میانگین. مثل یک تابع گاوسی ساده. 
حالا باید چیکار کنیم؟ 

```python
# standard scaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit_transform(train_features_numeric)
```

که خب خروجی زیر رو میده:
```txt 
array([[ 0.73708666, -0.81358055, 1.07200297, ..., -0.76459235, -0.09550628, 0.73694555], [ 0.76713228, -0.8744934 , -0.28020148, ..., 0.73787321, 0.99191205, 0.76695574], [-1.14577224, 0.74672556, -0.59836723, ..., 0.76351256, 2.03889292, -1.14610699], ..., [-1.26595472, 0.97632015, 0.5152129 , ..., -0.34667104, 0.67846881, -1.26616277], [ 0.10612862, 0.32502121, -1.47332306, ..., 0.5430142 , -0.02352337, 0.10658132], [-1.19084067, 0.63895667, -1.39378162, ..., 0.88401747, 0.22328275, -1.1913353 ]], shape=(16512, 9))
```

توی این حالت دیگه مینیمم لزوما منهای یک یا یک بازه مشخص نیست. 
نکته مهم اینه که عموما این استاندارد اسکیلر موقعی بهترین نتیجه و خروجی رو داره که توزیع داده نرمال باشه. ولی اگه خود داده توزیعی بسیار متفاوت از یک توزیع نرمال داشته باشه دیگه شرایط به این خوبی نخواهد بود. 

در این مواقع باید دوباره از تبدیلاتی استفاده بشه که بتونیم این توزیع ها رو شبیه تر کنیم به یک توزیع نرمال. مثلا چه تبدیلی میشه زد؟ 
مثلا بیایم برای یک فیچر که خیلی ناگاوسی هست بیایم جذر بگیریم. اینجوری اختلافشون خیلی کمتر میشه. یا لگاریتم بگیریم. 

### Data Transformation

بیایم یه ‍`log transformation` ها رو روی یکی از فیچرهامون بزنیم ببینیم چی میشه. 

```python
train_features_numeric["total_rooms"].hist(bins = 30 )
train_features_numeric["total_rooms"].apply(np.log).hist(bins = 30)
```
که نتایج آن در صفحه ژوپتیر مربوطه موجود است.  روی این داده لاگ نرمال جواب بهتری به ما میده به طور و به لحاظ ریاضی. 

برای ستون های `multimodal` کار های  متفاوتی میشه انجام داد. مثلا میشه از همون متد `pd.cut` که پیش تر استفادع کردیم و بازه بازه کردیم یک متغیر پیوسته رو میتونیم استفاده کنیم. در واقع داریم باکت باکت میکنیم اما باید در نظر داشت که هر باکت یک کتگوری به وجود میاره که بعد باز باید تبدیل به یک عدد بشه. 

کار دیگه ای که میشه انجام داد اینه که مثلا برای `housing_median_age` میشه یک ترنفسورمیشن روی یک توضیع مالتیمودال انجام میدیم که یک فیچر بیاد اضافه بشه بهش یا مثلا یک مد ماکسیمم رو درنظر میگیریم براش. حالا میایم فیچری که اضافه میکنیم فاصله بقیه نقاط از این مدماکسیمم رو نشون میده که بهش میگن similarity measure  که میاد بهمون شباهت بقیه نقاط رو از اون مد ماکسیمم رو بهمون میده. 
این شباهت رو با ابزاری محاسبه میکنیم به اسم `radial basis function` یعنی تابعی که فقط به فاصله اون مقدار ورودی و یک نقطه ثابت این کار رو برامون انجام میده که یکی از پر استفاده ترینشون همون توضیع گوسی یا نرمال هست. 
بیایم این کار رو برای housing_median_age انجام بدیم.
ابزاری که میشه استفاده کرد چیزی هست به اسم `rbf_kernel` که توضیحات زیر براش اومده:
Compute the rbf (gaussian) kernel between X and Y.
K(x, y) = exp(-gamma ||x-y||^2)
for each pair of rows x in X and y in Y.

ما میایم اون ستون خاص یا همون فیچری که این ویژگی رو داره از دیتافریممون مشخص میکنیم و جدا میکنیم. 
```python
from sklearn.metrics.pairwise import rbf_kernel
rbf_kernel(train_features_numeric[["housing_median_age"]], [[35]], gamma= 0.1)
```

بیایم اول رسمش کنیم تا یکم شهودی تر بشه. 
```python
fig, ax = plt.subplots()
ax.hist(train_features_numeric["housing_median_age"], bins=30)
plt.show()
```

مشکل اینجاست که `rbf_kernel` ورودی‌ها را به‌صورت آرایهٔ دوبعدی می‌خواهد، اما `np.linspace` یک آرایهٔ یک‌بعدی تولید می‌کند.

چون `housing_age` شامل ۱۰۰۰ نمونه از یک ویژگی است، باید آن را به شکل `(1000, 1)` تبدیل کنی:

```
housing_age = np.linspace(    train_features_numeric["housing_median_age"].min(),    train_features_numeric["housing_median_age"].max(),    1000)housing_age_rbf = rbf_kernel(    housing_age.reshape(-1, 1),    [[35]],    gamma=0.1)
```

چرا `reshape(-1, 1)`؟
- `housing_age` قبل از تغییر شکل: `(1000,)`
- بعد از تغییر شکل: `(1000, 1)`
- `[[35]]` هم شکل `(1, 1)` دارد.
در نتیجه، خروجی `housing_age_rbf` یک آرایه با ابعاد `(1000, 1)` است که مقدار شباهت RBF هر سن را نسبت به مقدار ۳۵ محاسبه می‌کند.
اگر بخواهی نمودارش را رسم کنی:

```
plt.plot(housing_age, housing_age_rbf)plt.xlabel("Housing Median Age")plt.ylabel("RBF Similarity")plt.grid()plt.show()
```

نکته ای که هست اینه که ما دیتامون رو برای سهولت در تحلیل تبدیل کردیم به لگاریتمی پس تمامی نتایجی هم که به دست میاریم به صورت لگاریتمی هست. وقتی میخوایم نتایج رو برگردونیم و شبیه کنیم به دیتای اصلیمون باید اینورس اون لگاریتم رو بزنیم پس خیلی مهمه که اینورس هر عملی که انجام میشه رو بتونیم روش بزنیم. 

توی سایکیت لرن توی داکیومنت های هر متدی که داریم استفاده میکنیم یه متد `inverse_transform` هم وجود داره که بعد از پری پراسسینگ باید اعمال بشه. 

خب پس اگه با یک روشی روی یک فیچری ترنسفورمیشنی زده شده با این متد inverse_transform میشه برش گردوند به حالت قبلش که روی تارگت برامون خیلی مهمه چون میخوایم ببینیم این مقدار پیشبینی شده روی واحد اصلی چی هست. 

خب حالا بیایم اینکارو انجام بدیم و در کنارش پروسه ساختن یک مدل با استفاده از scikit-learn رو هم ببینیم. فعلا با مدل ساده linear regression رو به کار میبریم. 
 
```python
from sklearn.linear_model import LinearRegression
train_target.to_frame()
ss = StandardScaler()
target_values = ss.fit_transform(train_target.to_frame())
target_values
```
توی این مرحله اول میایم مدل رو با یک استاندارد اسکیلر ترنسفورم میکنیم. در واقع مقادیری که برای تارگت میخوایم رو ترنسفورم میکنیم. حالا از یک مدل که در اینجا رگرسیون خطی ساده بود استفاده میکنیم برای این که با این مدل دیتا رو فیت کنیم به مدل. 
```python 
model = LinearRegression()
model.fit(train_features_numeric[["median_income"]], target_values)
```
اینجا این فیت کردن درواقع یک نگاشت برای پیدا کردن یک تابعی هست که میتونه دیتای ما و تارگت ما رو توضیح بده. 
حالا میتونیم یک پردیکشن از روی خود مدل رگرسیون خطی به دست بیاریم به صورت :
```python 
prediction = model.predict(train_features_numeric[["median_income"]].iloc[:5])
```
که خب خروجیش به صورت زیر هست :
```txt
array([[-0.06592182], [ 0.68465284], [ 1.40731612], [ 0.64749857], [ 0.47476959]])
```
همونطور که میشه دید این مقادیرش اصلا شبیه به مقادیر اولیه نیست چرا که ترنسفورم پیدا کرده و باید برش گردونیم به حالت یا اینورسش کنیم.
```python
ss.inverse_transform(prediction)
```
که خب خالا خروجیش شبیه تره:
```txt
array([[198527.05762785], [284696.99725944], [367662.56434888], [280431.49087807], [260601.29011982]])
```

---
اما راه و روش راحت تری هست که معادل کار بالاست ولی یکم سر راست تر:
```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),transformer=StandardScaler())
prediction = model.predict(train_features_numeric[["median_income"]].iloc[:5])

prediction
```

### FunctionTransformation

بحث دیگه ای که اهمیت داره برای ماشین لرنینگ بحث ترنسفورمرهایی هست که خودمون میخوایم ویژگی هاش رو مشخص کنیم. یعنی یه جور ترنسفورمر سفارشی که خودمون ویژگی هاش رو داریم. خیلی سخت نیست و خیلی ساده از ابزاری در سایکیت لرن استفاده میکنیم به اسم `FunctionTransformer`
```python
from sklearn.preprocessing import FunctionTransformer
FunctionTransformer()
logarithmic_transformer = FunctionTransformer(np.log,inverse_func= np.exp)
logarithmic_transformer.transform(train_features_numeric[["population"]])
```

یکی از موارد مهم توی این فانکشن ترنسفورم بحث `args` هست. به صورت پیشفرض ما فرضی که داریم اینه که تنها یک ورودی داریم و اون ورودی همون ستون و آرایه ای هست که میخوایم روش ترنسفورمیشن رو انجام بدیم.  ولی لزوما اینطور نیست. اگر بخوایم به صورت دلخواه از اول خودمون بنویسیم ممکنه که چند تا ورودی و حتی ضریب داشته باشه. 
اون موقع باید از `kw_args` استفاده کنیم. 

به عنوان مثال ما پیشتر اومدیم از یک تابع استفاده کردیم به اسم `rbf_kernel` که خب خودش دو نوع ورودی داشته.
۱. [[35]]
۲. gamma = 0.1
حالا در نتیجه اگه بخوایم از این تابع توی فانکشن ترنسفورم استفاده کنیم باید دوتا ورودی این رو هم بهش بدیم. به جه شکل؟
```python
rbf_transformer = FunctionTransformer(rbf_kernel, kw_args={"Y": [[35.0]],"gamma": 0.1})

rbf_transformer.transform(train_features_numeric[["housing_median_age"]])
```

از چیزایی که باید بهش دقت کنیم وقتی داریم از توابع آماده استفاده میکنیم اینه که این تابع آماده روی چند تا ستون به چه شکل کار میکنه؟ اگه یه دیتا فریم بدیم بهش که ۴ تا ستون داره به عنوان ورودی حالا این تابع اصلی توی این فانشکن ترنسفورمر به چه شکل کار میکنه ؟  

یه مشکلی که هست اینه که تا الان چیزی که دیدیم اینه که این فانکشن ترنسفورمی که دیدیم صرفا داره پروسه ترنسفورم رو انجام میده. به عبارتی دیگه صرفا داره یه تابع دلخواه رو اعمال میکنه  روی ستونی که ما داریم. اما یه جاهعایی ممکنه ما ترنسوفرمی رو بخوایم که قرار باشه یه سری پارامتر رو یاد بگیره مثل اونایی که رو داده ترین فیت میشدن و بعد روی داده تست ترنسفورم میکردیم. 
اینو چجوری میشه نوشت؟
سایکیت لرن کلاس ها  و آبجکت عها رو با اپروچی انجام میده که بهش میگن `duck typing`. منظور چیه؟ 
اگه چیزی شبیه به اوردک راه میره و شبیه به اوردک صدا میده پس باید اوردک باشه. توی داک تایپینگ کلاسی که میسازیم آبجکت های اون کلاس نوعشون چک نمیشه بلکه متدهاشون چک میشه(راه رفتن اوردک) 
یعنی درنتیجه اگه بخوایم چنین ترنسفورمی بسازیم باید متد های fit, transform, fit transformرو داشته باشه. 

### Custom Standard Scaler

```python
class CustomStandardScaler:
	def fit(self, X,y= None):
		self.mean_of_X = X.mean(axis = 0)
		self.std_of_X = X.std(axis = 0)
		# X.shape[0]: تعداد سطرها
		# X.shape[1]: تعداد فیچرها
		self.n_features_in_ = X.shape[1]
		return self

  

	def transform(self,X):
		assert self.n_features_in_ == X.shape[1]
		# asser: if above is not always true, it will give us an error
		X = X - self.mean_of_X
		return X / self.std_of_X
```

یه سری ویژگی هایی سایکیت لرن داره که اجازه میده کمتر کد بنویسیم:

```python


from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    # با ارث‌بری از این دو کلاس، Transformer ما با ساختار استاندارد Scikit-Learn سازگار می‌شود.
    # BaseEstimator: امکاناتی مثل get_params و set_params را فراهم می‌کند.
    # TransformerMixin: متد fit_transform را در اختیار ما قرار می‌دهد.

    def __init__(self, a=0):
        # متد سازنده کلاس؛ برای تعریف پارامترهای اولیه (Hyperparameters)
        self.hp = a

    def fit(self, X, y=None):
        # در این متد، اطلاعات آماری داده‌های ورودی را یاد می‌گیریم.
        # y در اینجا استفاده نمی‌شود، اما برای سازگاری با Scikit-Learn وجود دارد.

        X = check_array(X)
        # بررسی اعتبار داده ورودی و تبدیل آن به آرایه NumPy

        self.mean_of_X = X.mean(axis=0)
        # محاسبه میانگین هر ویژگی (هر ستون)
        # axis=0 یعنی محاسبات به‌صورت ستونی انجام شوند.

        self.std_of_X = X.std(axis=0)
        # محاسبه انحراف معیار هر ویژگی (هر ستون)

        # X.shape[0]: تعداد نمونه‌ها (سطرها)
        # X.shape[1]: تعداد ویژگی‌ها (ستون‌ها)

        self.n_features_in_ = X.shape[1]
        # ذخیره تعداد ویژگی‌های داده آموزشی
        # در transform از این مقدار برای بررسی تعداد ویژگی‌های داده جدید استفاده می‌کنیم.

        return self
        # برگرداندن خود شیء برای امکان استفاده از متدهای زنجیره‌ای


    def transform(self, X):
        # در این متد، داده‌ها را با استفاده از اطلاعاتی که در fit یاد گرفته‌ایم، استاندارد می‌کنیم.

        check_is_fitted(self)
        # بررسی می‌کند که آیا متد fit قبلاً اجرا شده است یا خیر.
        # اگر fit اجرا نشده باشد، خطا برمی‌گرداند.

        X = check_array(X)
        # بررسی اعتبار داده ورودی

        assert self.n_features_in_ == X.shape[1]
        # بررسی یکسان بودن تعداد ویژگی‌های داده جدید با داده آموزشی
        # assert: اگر شرط برقرار نباشد، خطای AssertionError ایجاد می‌کند.

        X = X - self.mean_of_X
        # کم کردن میانگین هر ویژگی از مقادیر همان ویژگی

        return X / self.std_of_X
        # تقسیم مقادیر حاصل بر انحراف معیار هر ویژگی
        # فرمول استانداردسازی:
        # X_scaled = (X - mean) / std
```

هر چند الان داره کار میکنه ولی هنوز کامل نیست. 

```python 
css = CustomStandardScaler()
css.fit(train_features_numeric)
css.transform(train_features_numeric)
# or 
css.fit_transform(train_features_numeric)
```

تا الان هزار تا کار انجام دادیم فقط برای Transformation. گیج کننده هست برای همین از یک پایپلاین منطقی استفاده میکنیم.

### Transformation Pipeline

برای اینکه یه سری پایپلاین دلخواه و مشخص رو بتونیم درست کنیم باید از متد `Pipeline` استفاده کنیم. 

```python

# Transformation pipeline
# ساخت یک Pipeline برای انجام چند مرحله پردازش متوالی روی داده‌های عددی

from sklearn.pipeline import Pipeline
numerical_features_pipeline = Pipeline([
    # مرحله اول: جایگزینی مقادیر خالی (NaN)
    ("medianImputer", SimpleImputer(strategy="median")),
    # در این مرحله، مقادیر خالی هر ویژگی با میانه (Median) همان ویژگی جایگزین می‌شوند.
    # Median نسبت به Outlierها حساسیت کمتری نسبت به Mean دارد.
    # مرحله دوم: نرمال‌سازی داده‌ها
    ("MinMaxScaler", MinMaxScaler()),
    # در این مرحله، مقادیر هر ویژگی به بازه پیش‌فرض [0, 1] منتقل می‌شوند.
    # فرمول MinMaxScaler:
    # X_scaled = (X - X_min) / (X_max - X_min)

])
# Pipeline مراحل بالا را به ترتیب اجرا می‌کند.
# ابتدا مقادیر خالی را جایگزین می‌کند و سپس داده‌ها را نرمال‌سازی می‌کند.
```

ولی نکته مهمی که باید در نظر بگیریم اینه که همه ترنفسفورمیشن هایی که توی پایپلاین میذاریم باید متد `fit_transform` رو داشته باشن. چرا؟
چون سایکیت لرن میاد به ترتیب توی این پایپلاین همه رو فیت ترنسفورم میکنه به جز اون متد آخر در نتیجه اگه ترنسفورمیشنی اون آخر قراره بمونه باید فیت ترنسفورم رو داشته باشه. اتفاقا این بد هم نیست. 
اگه قرار باشه یه مدل یادگیری ساخت خودمون هم توی این پایپلاین باشه الزاما باید جاش آخرین مرحله پایپلاین باشه که روش فیت ترنسفورم انجام نشه. 
در نهایت یه آبجکت از آخرین ترنسفورمر پایپلاین رو بهمون میده. 

حالا بیایم تستش هم بکنیم:
```python 
numerical_features_pipeline.fit_transform(train_features_numeric)
```
که خب خودش همه کار ها رو میکنه و خروجیش به صور ت زیر هست:

```txt
array([[0.62350598, 0.1434644 , 0.80392157, ..., 0.03322368, 0.21983145, 0.6232378 ], [0.62948207, 0.12964931, 0.47058824, ..., 0.12960526, 0.36151915, 0.62920473], [0.24900398, 0.49734325, 0.39215686, ..., 0.13125 , 0.49793796, 0.24883068], ..., [0.2250996 , 0.54941552, 0.66666667, ..., 0.06003289, 0.32067834, 0.22496001], [0.49800797, 0.40170032, 0.17647059, ..., 0.11710526, 0.22921063, 0.49790256], [0.24003984, 0.47290117, 0.19607843, ..., 0.13898026, 0.26136881, 0.23983795]], shape=(16512, 9))
```

### Column Transformer 

تا الان داده رو دوقسمت کردیم:
۱. کتگوری
۲. نومریک
درنتیجه دو نوع ترنسفورمیشن هم برای اینا داشتیم:
۱. کتگوریکال
۲.نومریکال

میشه کاری کرد که جفتش با هم انجام بشه:

```python 
from sklearn.compose import ColumnTransformer
```

کاری که میشه کرد اینه که اول بیایم اسم ستون ها رو مشخص کنیم.  

```python
train_features.columns
```

که خب خروجی زیر رو میده :

```txt
Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'identifier'], dtype='str')
```

حالا میتونیم اینو تبدیل کنیم به دوتا لیست نومریک و کتگوریکال که مشخص تر بشه همه چی. 

```python 
numerical_features = list(train_features.columns)
numerical_features
numerical_features.remove("ocean_proximity")
numerical_features
```

که خب خروجیش به صورت زیر در میاد:
```python
['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'identifier']
```

و همینکارو برای کتگوریکال ها در بیارم:

```python
categorical_features = ["ocean_proximity"]
categorical_features
```

حالا مشابه همون پایپلاینی که بالاتر برای نومریکال فیچر ها نوشتم رو برای کتگوریکال ها هم بنویسم:

```python 
categorical_features_pipeline = Pipeline([
	("mfImputer",SimpleImputer(strategy="most_frequent")),
	("ohe",OneHotEncoder())
])
```

که خب اینجا مجبور شدیم جدا کنیم. برای اینکه اینکارا رو نخوایم انجام بدیم خیلی ساده تر میتونیم بیایم  و یک جا جفتشو انجام بدیم به صورت زیر:

```python 
total_transformation = ColumnTransformer([
	("numerical", numerical_features_pipeline, numerical_features),
	("categorical", categorical_features_pipeline, categorical_features),
])
```

خب حالا میشه این ترنسفوریشن کلی رو روی یک دیتاست اعمال کنیم (درواقع فیت ترنسفورم کنیم) 
برای مثال روی همون train_features خواهیم داشت:
```python
total_transformation.fit_transform(train_features)
```

که خب خروجی زیر رو میده :
```txt
array([[0.62350598, 0.1434644 , 0.80392157, ..., 0. , 0. , 0. ], [0.62948207, 0.12964931, 0.47058824, ..., 0. , 0. , 0. ], [0.24900398, 0.49734325, 0.39215686, ..., 0. , 0. , 0. ], ..., [0.2250996 , 0.54941552, 0.66666667, ..., 0. , 1. , 0. ], [0.49800797, 0.40170032, 0.17647059, ..., 0. , 0. , 0. ], [0.24003984, 0.47290117, 0.19607843, ..., 0. , 0. , 1. ]], shape=(16512, 14))
```

که خب هموطنور که دیده میشه اینا صرفا خروجی عددی اینا هست در صورتی که ما اسم های فیچر ها رو هم میخوایم که برای اینکار باید از متد `.get_features_name_out()` استفاده کنیم. 

```python
total_transformation.get_feature_names_out()
```
که خروجی زیر رو میده بهمون :

```txt 
array(['numerical__longitude', 'numerical__latitude', 'numerical__housing_median_age', 'numerical__total_rooms', 'numerical__total_bedrooms', 'numerical__population', 'numerical__households', 'numerical__median_income', 'numerical__identifier', 'categorical__ocean_proximity_<1H OCEAN', 'categorical__ocean_proximity_INLAND', 'categorical__ocean_proximity_ISLAND', 'categorical__ocean_proximity_NEAR BAY', 'categorical__ocean_proximity_NEAR OCEAN'], dtype=object)
```

### Pipeline for transforming columns 

اما کاری که ما باید انجام بدیم درواقع پیدا کردن ترنسفورمیشن ها بر اساس دیتا ست و فیچرهامون هست که باید اینکارو انجام بدیم. 

یه موقع هایی هست که ما اصلا نیاز نداریم و یا نمیخوایم هیچ نوع ترنسفورمیشنی انجام بدیم روی یک سری ستون ها. برای اینکار میتوتیم از `drop` استفاده کنیم. در این صورت اون پایپلاین میاد اون ستون ها یا فیچرهایی که توی دراپ هستند رو حدف میکنه و دیگه روی اون ها ترنسفورمیشن رو اعمال نمیکنه چون چیزی نیست که روش اعمال بشه چرا که حذف شده. 

یه موقعی هست که نمیخوایم حذف بشه صرفا میخوایم از روش بدون تغییر رد بشه. اون موقع باید از `passthrough` استفاده کنیم.

بیایم از اول تغییرات رو روی داده ورودی فایل سی اس وی شروع کنیم.
بیایم اول مقدمات رو بچینیم و برای یک سری از فیچرهامون که مثلا تیل سنگینی دارن یک سری پایپلاین و ترنفورمیشن بنویسیم:

```python 
from sklearn.pipeline import make_pipeline
	heavy_tail_pipeline = make_pipeline(
	SimpleImputer(strategy="median"),
	FunctionTransformer(np.log,feature_names_out="one-to-one"),
	StandardScaler()
)
```

دوباره  مثل قبل برای کتگوریکال ها و نومریکال ها این کار ها رو انجام میدیم:
```python
categorical_features_pipeline = Pipeline([
	("mfImuper", SimpleImputer(strategy="most_frequent")),
	("oho", OneHotEncoder())
])
```

```python
housing_median_age = rbf_transformer = FunctionTransformer(rbf_kernel,
					kw_args={
					"Y": [[35.0]],
					"gamma": 0.1
})

housing_median_age = rbf_transformer.transform(train_features_numeric[["housing_median_age"]])
```

```python
default_numerical_pipline = make_pipeline(
	SimpleImputer(strategy="median"),
	StandardScaler()
)
```

حالا یک سری تابع بنویسیم برای عملیات هایی به صورت مستقیم روی فیچرها و ستون هامون. که لازم نباشه دستی اینکارو بکنیم و صرفا یک بار با نوشتن تابعش اینکارو انجام بدیم:

```python
def ratio_of_columns(X):
    # محاسبه نسبت ستون اول به ستون دوم
    # X[:, [0]] ستون اول و X[:, [1]] ستون دوم را انتخاب می‌کند.
    # استفاده از براکت دوبل باعث حفظ ساختار دوبعدی آرایه می‌شود.
    return X[:,[0]] / X[:,[1]]


def ratio_of_columns_name(function_transformer, fitures_names):
    # تعیین نام ویژگی جدیدی که توسط ترنسفورمر ایجاد می‌شود.
    # در اینجا نام ستون خروجی را "ratio" قرار می‌دهیم.
    return ["ratio"]


def ratio_of_columns_pipeline():
    # ساخت یک Pipeline برای اجرای مراحل پیش‌پردازش به ترتیب

    return make_pipeline(
        # جایگزینی مقادیر گمشده (NaN) با میانه هر ستون
        SimpleImputer(strategy="median"),

        # اعمال تابع ratio_of_columns برای محاسبه نسبت دو ستون
        # feature_names_out مشخص می‌کند نام ویژگی خروجی چگونه تعیین شود.
        FunctionTransformer(ratio_of_columns, feature_names_out= ratio_of_columns_name),

        # استانداردسازی ویژگی خروجی برای قرار گرفتن در مقیاس مناسب
        # میانگین داده‌ها را به صفر و انحراف معیار را به یک نزدیک می‌کند.
        StandardScaler()
    )
```

حالا میتونیم بیایم بگیم که تمام قدم های ترنفورمیشن رو اینجا میخوایم انجام بدیم:

```python
# ترکیب تمام مراحل پیش‌پردازش ویژگی‌ها در یک ColumnTransformer
all_transformations = ColumnTransformer([

    # محاسبه نسبت تعداد اتاق‌خواب‌ها به تعداد کل اتاق‌ها
    ("bedrooms", ratio_of_columns_pipeline(), ["total_bedrooms", "total_rooms"]),

    # محاسبه میانگین تعداد اتاق‌ها به ازای هر خانوار
    ("rooms_per_house", ratio_of_columns_pipeline(), ["total_rooms", "households"]),

    # محاسبه تعداد افراد به ازای هر خانوار
    ("peaple_perhouse", ratio_of_columns_pipeline(), ["population","households"]),

    # اعمال تبدیل لگاریتمی روی ویژگی‌هایی با توزیع دنباله‌سنگین (Heavy-tailed)
    # heavy_tail_pipeline از قبل ساخته شده و نیازی به فراخوانی با () ندارد.
    ("log", heavy_tail_pipeline, ["total_rooms","total_berdrooms", "population", "housholds", "median_income"]),

    # اعمال تبدیل RBF روی ویژگی سن خانه‌ها
    ("rbf", housing_median_age, ["housing_median_age"]),

    # پردازش ویژگی‌های دسته‌ای با استفاده از Pipeline مربوط به داده‌های categorical
    ("cat", categorical_features_pipeline, ["ocean_proximity"])

# هر ویژگی عددی دیگری که در مراحل بالا مشخص نشده باشد،
# با استفاده از default_numerical_pipline پردازش می‌شود.
], remainder=default_numerical_pipline)
```

حالا ترنسفورمیشن هامون آماده هست ولی کاری که باید بکنیم اینه که دیتاست های train و test و تارگت رو ایجاد کنیم:

```python 
train_features = train.drop(columns="median_house_value")
train_target = train["median_house_value"]
```

```python
test_features = test_set.drop(columns="median_house_value")
test_target = test_set.drop(columns="median_house_value")
```

حالا میتونیم اون پایپلاین همه ترنسفورمیشن ها رو روی این دیتاست هامون اعمال کنیم.

```python
train_features_transformed = all_transformations.fit_transform(train_features)
train_features_transformed.shape
all_transformations.get_feature_names_out()
```

که خب حالا خروجی زیر رو میده برای ستون هایی که داشتیم برای دیتا فریممون:
```txt
array(['bedrooms__ratio', 'rooms_per_house__ratio', 'peaple_perhouse__ratio', 'log__total_rooms', 'log__total_bedrooms', 'log__population', 'log__households', 'log__median_income', 'rbf__housing_median_age', 'cat__ocean_proximity_<1H OCEAN', 'cat__ocean_proximity_INLAND', 'cat__ocean_proximity_ISLAND', 'cat__ocean_proximity_NEAR BAY', 'cat__ocean_proximity_NEAR OCEAN', 'remainder__longitude', 'remainder__latitude', 'remainder__identifier'], dtype=object)
```


### انتخاب و آموزش مدل

تا الان داده رو مرتب و تمیز کردیم. حالا باید بریم مدل رو انتخاب کنیم. 
ساده ترین مدل های یادگیری ماشین مدل های خطی هستند. مهم ترین بخشی که در سایکیت لرن وجود داره برای مدل های مختلف مثلا همین مدل خطی `linear regression` متدهاش هستن. مثلا:
- fit
- get_params 
- predict
- score
- set_params

پس فعلا که میخوایم این مدل خطی رو از سایکیت لرن استفاده کنیم اول میایم ایمپوریتش میکنیم:
```python 
from sklearn.linear_model import LinearRegression
```
که خب این یک سری دیتا بریا اینپوت لازم داره که ما تا اینجا داشتیم دیتایی که داشتیم رو تمیز میکردیم برای دقیقا همینکار.
نکته ای که هست اینه که ما حتی بخش مدل رو هم میتونیم بذاریم توی پایپلاینی که داریم. عملا یخ بخش به انتهای همون پایپلاینمون اضافه میشه
```python 
linear_regression_pipeline = make_pipeline(all_transformations, LinearRegression)
```

حالا میتونیم با `fit` اعمالش کنیم روی دیتامون:

```python
linear_regression_pipeline.fit(train_features, train_target)
```

حالا ساخت مدل برای پیشبینی کردن یعنی چی؟ یعنی از این پایپلاینی که استفاده کردیم با `predict` پیشبینی کنیم. برای پیشبینی کردن باید فیچر بهش بدیم. ولی نمیدونیم تارگت چیه و میخوایم مدل برامون پیشبینی کنه. 

```python 
house_value_prediction = linear_regression_pipeline.predict(train_features)
```
که خب خروجیش اینه :
```txt
array([214917.39423674, 282407.18018002, 343128.02871597, ..., 282441.18916614, 176719.97564541, 292691.08141043], shape=(16512,))
```

حالا بیایم با مقدار ترین تارگت مقایسه کنیم.

```python 
train_target.values
```

که خروجیش به صورت زیر هست:

```txt
array([171500., 221900., 362700., ..., 266400., 94600., 221700.], shape=(16512,))
```

ولی سوال اینه که چجوری عملکرد مدل رو روی داده های ترین ارزیابی کنیم؟

### معیارهای ارزیابی مدل رگرسیون 

سوال اینه که چجوری مدل رو تنظیم کنیم؟ معیار ارزیابی چیه؟ باید اینکارا رو انجام بدیم که بهترین عملکرد مدل رو بکشیم بیرون. 
لانچ و نگهداری مدل چیه؟
برای ارزیابی رگرسیون معیار های مختلفی میتونیم داشته باشیم:
**Mean Absolute Error (MAE)**
$$[ MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| $$و 

**Mean Squared Error (MSE)**
$$ MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 $$

- $(n$): تعداد نمونه‌ها
- $(y_i$): مقدار واقعی (Actual Value)
- $(\hat{y}_i$): مقدار پیش‌بینی‌شده (Predicted Value)
- $(y_i - \hat{y}_i)$: میزان خطا (Prediction Error)

حالا بیایم این دومعیار رو روی خروجی های مدلمون حساب کنیم ببینیم چی میشه. 

معیار بعدی ای که استفغاده میشه برای ارزیابی چیزی هست که به اسم $R^2$ میشناسیمش یا ضریب تعیین. که برای محاسبه اون به چند تا پیش نیاز دیگه نیاز مندیم.
**Mean (Arithmetic Mean)**
میانگین حسابی:

$$ \bar{y} = \frac{1}{n}\sum_{i=1}^{n}y_i $$

**توضیح نمادها:**

- $(n)$: تعداد داده‌ها
- $(y_i)$: مقدار هر داده
- $(\bar{y})$: میانگین مقادیر داده‌ها
و با استفاده از این مقدار میانگین میتوان مقادیر دیگری رو به دست‌آورد: 
#### **1. Sum of Squares of Residuals (SSR)**
مجموع مربعات باقی‌مانده‌ها:

$$ SS_R = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 $$
به نوعی یک معیاری از خطای داده هست.
#### $**2. Total Sum of Squares (SST)**
مجموع مربعات کل:

$$ SS_T = \sum_{i=1}^{n}(y_i - \bar{y})^2 $$
به نوعی وریشن داده رو نسبت به میانگین داده بهمون میده. 
### **3. Coefficient of Determination (R²)**
ضریب تعیین:
$$ R^2 = 1 - \frac{SS_R}{SS_T} $$

**توضیح نمادها:**

- $(y_i)$: مقدار واقعی (Actual Value)
- $(\hat{y}_i)$: مقدار پیش‌بینی‌شده (Predicted Value)
- $(\bar{y})$: میانگین مقادیر واقعی
- $(n)$: تعداد نمونه‌ها
- $(SS_R)$: مجموع مربعات خطاهای مدل (Residual Sum of Squares)
- $(SS_T)$: مجموع مربعات انحراف مقادیر واقعی از میانگین

**نکته:** هرچه$(R^2)$ به ۱ نزدیک‌تر باشد، مدل بخش بیشتری از تغییرات داده‌های هدف را توضیح می‌دهد.

فرق این معیار با سه تا معیار قبلی اینه که اون سه تای قبلی از جنس خط هستند اما این معیار از جنس امتیاز هست در نتیجه هدف ما اینه که اون سه معیار قبلی رو کمتر کنیم و این معیار جدید رو بیشتر کنیم. 
برای بالا بردن این امتیاز عملا باید کسر آخر رو کمترین مقدار کنیم و یا در حالت ایده آل صفر کنیم.

نکته: ممکنه که مقدار $(R^2)$ حتی منفی باشه یعنی اون کسر داخلش بزرگ تر از یک باشه. این ممکن هست ولی نشون میده که مدل خیلی بده. در نتیجه وقتی خطا بیشتر از واریانس باشه.

### ارزیابی مدل رگرسیون خطی 

حالا بیایم ببینیم این ها رو چجوری میشه توی سایکیت لرن پیاده سازی کنیم. در بخش متریک های سایکیت لرن میتونیم از انواع و اقسام متریک های سایکیت لرن استفاده کنیم.

```python
from sklearn.metrics import mean_absolute_error
# sklearn.metrics.mean_absolute_error(y_true, y_pred, *,
# sample_weight=None, multioutput='uniform_average')
# here y_true = train_target.values
# y_pred = house_value_prediction
mean_absolute_error(train_target.values, house_value_prediction)
```

که خروجیش یک عدد هست. برای ما شده 53453.957551597116. 
شاید بگیم چقدر زیاده. مگر نباید ارور کوچیک باشه. در واقع نه! ارور بستگی به رنج داده ما داره
برای معیار بعدیمون هم توان دو رو داشتیم.

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(train_target.values, house_value_prediction)
```

5085074260.047806

حالا میتونیم از این جذر هم بگیریم . برای اینکار از کتابخوانه دیگری اسنفاده میشه:

```python
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(train_target.values, house_value_prediction)
rmse
```

که خروجی میده: 71309.70663274254

حالا میتونیم اون اسکور رو هم حساب کنیم:

```python 
# or using its library
from sklearn.metrics import r2_score
r2_score(train_target.values, house_value_prediction)
```
0.6141899717089189

البته باید در نظر داشت که برای ارزیابی مدل نباید فقط به همین یک معیار و امتیاز بسنده کرد. نتایج ما نشون میده که این مدل خیلی قوی نیست و عملا داره آندرفیتینگ انجام میده. 
 سه تا آپشن داریم:
 ۱. فیچر بهتر اضافه کنیم.
 ۲. مدل رگیولاریزشن استفاده کنیم.
 ۳. مدل قوی تر بنویسیم
### مفهوم cross validation

از مدل هایی که که نسبت به مدل خطی قوی تر هست مدل های خانواده درخت تصمیم یا decision tree هست. 

برای استفاده از این مدل هم مثل سابق میتونیم خیلی ساده از یک پایپلاین استفاده کنیم. به شکل:

```python
from sklearn.tree import DecisionTreeRegressor
# using a piplelone for models
dt_regression_pipeline = make_pipeline(all_transformations, DecisionTreeRegressor())
```

حالا توانایی این رو داریم که این مدل رو فیت کنیم 

```python 
dtrp = dt_regression_pipeline.fit(train_features,train_target)
```
حالا میتونیم روی داده اعمالش کنیم ولی این بار به صورت مستقیم روی اون معیار های ارزیابیمون:

```python.
mean_absolute_error(train_target.values, dtrp.predict(train_features))
```

که خروجی خطاش صفره. بیایم اسکورش رو حساب کنیم. 

```python
r2_score(train_target.values, dtrp.predict(train_features))
```

گخ جوابش دقیقا یک هست. یعنی صد در صد مدل داره توضیح میده. مگه ممکنه؟ آره چون مدل اوور فیت شده. حالا اباید چیکار کنیم؟ 

باید یک بخشی از داده یادگیری یا اون بخش ولیدیشن فیچر ها رو نشون میدیم ولی تارگت ها رو نشون نمیدیم. اینجوری میتونیم متوجه بشیم ببینیم مدل روی اده هایی که ندیده چقدر خوب عمل میکنه. 

اما کاری که میتونیم بکنیم که موثر تر هست کاری هست که بهش میگن cross validation. در ان داده ما سه بخش میشه
train
validation
test

ایده هاش اینه که بیایم از داده های ترین و ولیدیشن بهتر استفاده کنیم. cv3 

توضیح کراس ولیدیشن 

 ### پیاده سازی cross validation 
 برای اینکار به صورت سخت اینه که اون ترین تست اسپلیت رو ۵ بار انجام بدیم و ۵ تا دسته بندی مختلف درست کنیم. رو هر کدوم یه مدل پیاده کنیم و اون ها رو ارزیابی کنیم. ۵ بار مثاله .

ولی کار راحت تر ی هست که بیایم از زکتابخونه خود سایکیت لرن استفاده کنیم. `cross_val_score` که خودش این کار رو با استیمیتور انجام میده. 
```python
from sklearn.model_selection import cross_val_score
dtrp_cross = -cross_val_score(dtrp, train_features, train_target, scoring= "neg_mean_absolute_error", cv = 5 )
```

اینجوری به طور خودکار میاد با معیاری به اسم `neg_mean_absolute_error` کراس ولیدیشن رو انجام میده. میتونیم خروجیش رو تبدیل کنیم به یک دیتا فریم و دیسکرایبش کنیم. خیلی راحت:

```python
pd.Series(dtrp_cross).describe()
```

که خروجی زیر رو میده :
```txt
count 5.000000 
mean 45213.528467 
std 760.938546 
min 44273.638208 
25% 44559.378444 
50% 45420.515748 
75% 45906.009388 
max 45908.100545 
dtype: float64
```

### underfit-overfit

حالا که یه مدل قوی تر رو دیدیم. ولی باید بسیار مراقب اورفیتینگ باشیم. برای یک مدل میشه گراف ارور بر حسب پیچیدگی رو رسم کرد. همچین نموداری به طور ساده بهمون میگه هر چه مدل پیچیدگی و پارامتر های بیشتری داشته باشه خطای train کمتر و کمتر میشه. اما هرچقدر پچیدگی رو بیشتر میکنیم خطای validation error از یه جایی شروع میکنه به بیشتر شدن. 

هدف اصلی ما هم بهترین ولیدیشن هست نه ترین پس باید مینیمم نمودار cross validation رو پیدا کنیم. حالا از روی منحتی ترین و ولیدیشن دو ناحیه داریم:
- کم برازش underfitting: جایی که هم train و هم validation ارور خیلی زیادی دارند. 
- بیش برازش overfitting: جایی که train خطای بسیار کمی داره و validation ارور زیادی داره. 
به طور کلی ما به دنبال تغییر هایپرپارامتر ها هستیم برای پیدا کردن بهترین اسکور یا کمترین ارور روی ولیدیشن


تا اینجا ما معیار ارزیابی رو میخواستیم برای تنظیم مدل. حالا میخواین مدل رو تنظیم کنیم. یعنی چی؟ یعنی اگر هایپر پارامترهایی وجود داره بیایم بهترین هایپرپارامتر ها رو پیدا کنیم که ارور ها کمترین مقدار خود رو داشته باشن. 

حالا باید این هایپرپارامتر ها رو بهینه سازی کنیم.

### بهینه کردن هایپرپارامترها 


برای تنظیم کردن هایپرپارامتر ها از یکی از امکانات سایکیت لرن میشه استفاده کرد به اسم GridSearchCV. 

```python
class sklearn.model_selection.GridSearchCV(_estimator_, _param_grid_, _*_, _scoring=None_, _n_jobs=None_, _refit=True_, _cv=None_, _verbose=0_, _pre_dispatch='2*n_jobs'_, _error_score=nan_, _return_train_score=False_)
```

این ابزار هم در ابتدا خیلی شبیه مدل کار میکنه و یه چیز جنرال هست. یعنی هم میتونه برای مدل ها باشه و هم ترنسفورما و بعد از خروجی اون یک سری اطلاعات رو بگیریم مثلا best_params

```python
from sklearn.model_selection import GridSearchCV

# تعریف پارامترهایی که قرار است آزمایش شوند
param_grid = {
    "max_depth": [5, 10, 15]
}

# ایجاد GridSearchCV برای پیدا کردن بهترین پارامتر
gs = GridSearchCV(
    estimator=DecisionTreeRegressor(),
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    error_score="raise"
)

# اجرای Grid Search
gs.fit(train_features_transformed, train_target)

# نمایش بهترین پارامترها و امتیاز
print("Best Parameters:", gs.best_params_)
print("Best MAE:", -gs.best_score_)
```

که خروجی زیر رو میده :
```python
{'max_depth': 10}
```


یه متد دیگه ای که هست اینه که از پایپلاین استفاده کنیم دوباره 

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ساخت Pipeline شامل مراحل پیش‌پردازش و مدل Decision Tree
dt_regression_pipeline = Pipeline([
    ("transformation", all_transformations),
    ("dtr", DecisionTreeRegressor())
])

# تعریف پارامترهایی که قرار است توسط GridSearchCV بررسی شوند
param_grid = {
    # بررسی دو مقدار مختلف برای پارامترهای RBF
    "transformation__rbf__kw_args": [
        {"Y": [[35.0]], "gamma": 0.1},
        {"Y": [[35.0]], "gamma": 0.2}
    ],

    # بررسی عمق‌های مختلف برای درخت تصمیم
    "dtr__max_depth": [5, 10, 15]
}

# جست‌وجوی بهترین ترکیب پارامترها با Cross Validation
gs = GridSearchCV(
    estimator=dt_regression_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    error_score="raise"
)

# آموزش و ارزیابی مدل
gs.fit(train_features, train_target)

# نمایش بهترین نتایج
print("Best Parameters:", gs.best_params_)
print("Best MAE:", -gs.best_score_)
```

که خروجیش به صورت زیر هست 
```txt
Best Parameters: {'dtr__max_depth': 10, 'transformation__rbf__kw_args': {'Y': [[35.0]], 'gamma': 0.1}} Best MAE: 40658.99582386962
```

کل نتایج رو میشه به صورت زیر به دست آورد:

```python
gs.cv_results_
```

```txt
{'mean_fit_time': array([0.27971244, 0.20300002, 0.28591633, 0.26428657, 0.35339108, 0.33485789]), 'std_fit_time': array([0.1030603 , 0.01597551, 0.02346618, 0.01573587, 0.02122178, 0.01660955]), 'mean_score_time': array([0.03400979, 0.02645035, 0.02467246, 0.02308302, 0.02617826, 0.02487931]), 'std_score_time': array([0.01094037, 0.00337813, 0.00194716, 0.00089195, 0.0023555 , 0.0016338 ]), 'param_dtr__max_depth': masked_array(data=[5, 5, 10, 10, 15, 15], mask=[False, False, False, False, False, False], fill_value=999999), 'param_transformation__rbf__kw_args': masked_array(data=[{'Y': [[35.0]], 'gamma': 0.1}, {'Y': [[35.0]], 'gamma': 0.2}, {'Y': [[35.0]], 'gamma': 0.1}, {'Y': [[35.0]], 'gamma': 0.2}, {'Y': [[35.0]], 'gamma': 0.1}, {'Y': [[35.0]], 'gamma': 0.2}], mask=[False, False, False, False, False, False], fill_value=np.str_('?'), dtype=object), 'params': [{'dtr__max_depth': 5, 'transformation__rbf__kw_args': {'Y': [[35.0]], 'gamma': 0.1}}, {'dtr__max_depth': 5, 'transformation__rbf__kw_args': {'Y': [[35.0]], 'gamma': 0.2}}, {'dtr__max_depth': 10,

...

'mean_test_score': array([-48296.72887937, -48296.88060797, -40658.99582387, -41119.4312364 , -42328.82925824, -42423.85097693]), 'std_test_score': array([797.03735267, 797.10502713, 718.11029835, 730.01584721, 798.47960776, 709.03082354]), 'rank_test_score': array([5, 6, 1, 2, 3, 4], dtype=int32)}

Output is truncated. View as a [scrollable element](command:cellOutput.enableScrolling?f7a73c75-fb1a-4d6b-83f2-887240da3300) or open in a [text editor](command:workbench.action.openLargeOutput?f7a73c75-fb1a-4d6b-83f2-887240da3300). Adjust cell output [settings](command:workbench.action.openSettings?%5B%22%40tag%3AnotebookOutputLayout%22%5D)...
```

یه مشکلی که این متد داره اینه که باید مقادیر ورودی مثل `"dtr__max_depth": [5, 10, 15]`  رو خودمون دستی وارد کنیم. که خب این هم میتونه سخت باشه و هم میتونه کند کنه یکم. برای اینکار میتونیم یک فضای بزرگ تر رو سرچ کنیم ولی به صورت رندوم . پس از قابلیت `randomize serach ` استفاده میکنیم. 

یا به صورت کلاسیک تر از کتابخونه scipy استفاده کنیم. برای اینکار یه بازه بهش میدیم و با یک تابع توزیع مشخصی میاد عدد رندوم میده.

```python
from scipy.stats import randint 
randint(low=5,high=100)
```

ولی خب برگردیم به همون رندومایز سرچ

```python
housing_model = gs.best_estimator_
housing_model["dtr"].feature_importances_.round(3)
```

```txt
array([0.011, 0.015, 0.114, 0.007, 0.003, 0.004, 0.003, 0.543, 0.014, 0.001, 0.169, 0. , 0. , 0.002, 0.022, 0.047, 0.043])
```


```python
# مرتب‌سازی ویژگی‌ها بر اساس میزان اهمیت در مدل Decision Tree

sorted(

zip(

# دریافت میزان اهمیت هر ویژگی و گرد کردن آن تا سه رقم اعشار

housing_model["dtr"].feature_importances_.round(3),

  

# دریافت نام ویژگی‌ها پس از انجام مراحل پیش‌پردازش

housing_model["transformation"].get_feature_names_out()

),

# مرتب‌سازی به صورت نزولی (از بیشترین اهمیت به کمترین اهمیت)

reverse=True

)
```

```txt
[(np.float64(0.543), 'log__median_income'), (np.float64(0.169), 'cat__ocean_proximity_INLAND'), (np.float64(0.114), 'peaple_perhouse__ratio'), (np.float64(0.047), 'remainder__latitude'), (np.float64(0.043), 'remainder__identifier'), (np.float64(0.022), 'remainder__longitude'), (np.float64(0.015), 'rooms_per_house__ratio'), (np.float64(0.014), 'rbf__housing_median_age'), (np.float64(0.011), 'bedrooms__ratio'), (np.float64(0.007), 'log__total_rooms'), (np.float64(0.004), 'log__population'), (np.float64(0.003), 'log__total_bedrooms'), (np.float64(0.003), 'log__households'), (np.float64(0.002), 'cat__ocean_proximity_NEAR OCEAN'), (np.float64(0.001), 'cat__ocean_proximity_<1H OCEAN'), (np.float64(0.0), 'cat__ocean_proximity_NEAR BAY'), (np.float64(0.0), 'cat__ocean_proximity_ISLAND')]
```

این کد میزان اهمیت (Feature Importance) هر ویژگی را در مدل Decision Tree محاسبه و آن‌ها را از بیشترین اهمیت به کمترین اهمیت مرتب می‌کند. به این ترتیب می‌توانیم تشخیص دهیم کدام ویژگی‌ها بیشترین نقش را در پیش‌بینی مدل دارند و کدام ویژگی‌ها تأثیر کمتری دارند.

به ترتیب از بالا فیچر ها اهمیت مهم تری دارن برای مدلمون. 

### Confidence interval
**مفهومی:**

Confidence Interval بازه‌ای است که با یک سطح اطمینان مشخص، مقدار واقعی پارامتر جامعه (مانند میانگین جامعه) را تخمین می‌زند.

مثلاً فاصله اطمینان ۹۵٪ یعنی اگر نمونه‌گیری و محاسبه این بازه را بارها تکرار کنیم، تقریباً ۹۵٪ از بازه‌های به‌دست‌آمده پارامتر واقعی جامعه را پوشش خواهند داد.

### **فرمول ریاضی**

برای فاصله اطمینان میانگین جامعه، وقتی انحراف معیار جامعه نامعلوم است:

\[ CI = \bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}} \]

**اگر انحراف معیار جامعه معلوم باشد:**

\[ CI = \bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \]

### **توضیح نمادها**

- \(\bar{x}\): میانگین نمونه (Sample Mean)
    
- \(s\): انحراف معیار نمونه (Sample Standard Deviation)
    
- \(\sigma\): انحراف معیار جامعه
    
- \(n\): تعداد نمونه‌ها
    
- \(t\): مقدار بحرانی توزیع t
    
- \(z\): مقدار بحرانی توزیع نرمال استاندارد
    

### **مثال**

فرض کن میانگین درآمد یک نمونه برابر ۵۰ میلیون تومان و فاصله اطمینان ۹۵٪ آن به شکل زیر باشد:

\[ CI = [45,\ 55] \]

یعنی بر اساس روش نمونه‌گیری و محاسبات، بازه تخمینی ما برای میانگین واقعی جامعه بین ۴۵ تا ۵۵ میلیون تومان است.

**نکته مهم:** فاصله اطمینان ۹۵٪ به این معنی نیست که احتمال ۹۵٪ وجود دارد پارامتر واقعی در همین بازه مشخص باشد؛ بلکه به ویژگی آماری روش ساخت بازه اشاره دارد.

حالا بیایم از سای پای استفاده کنیم برای محاسبه این کانفیدنس اینتروال

```python
from scipy import stats
import numpy as np

ci = 0.95

# محاسبه مربع خطای هر نمونه
se = (test_target.values - housing_model.predict(test_features)) ** 2

# محاسبه بازه اطمینان RMSE
rmse_ci = np.sqrt(
    stats.t.interval(
        ci,
        len(se) - 1,
        loc=np.mean(se),
        scale=stats.sem(se)
    )
)

print(rmse_ci)
```

### اقدامات قبل از نهایی کردن مدل

نکته:‌ به فرض ما بخوایم این مدل رو استفاده کنیم برای یک تحلیل یا اپلیکیشن یا هر چیز دیگه. اولا که ما نمیایم کد رو کامل بفرستیم. 
به جاش میتونیم فایل مدلش رو بفرستیم. یعنی مدل ترین شده اش رو: 

```python
import joblib
joblib.dump(housing_model,"housing_model.pkl")
```

بعد از این فایل مدل در دایرکتونری پروژه با اسم ['housing_model.pkl'] سیو میشه. ولی در جای دیگه باید همه کتابخانه های لازم ایمپورت بشه . بعد فایل رو فراخوانی کنیم و استفاده کنیم.

حالا چجوری فراخوانی کنیم؟
```python
housing_model_loaded = joblib.load("housing_model.pkl")
```

یه نکته ای که هست برای بعد از انتشار مدل هست. مدل شاید لازم داشته باشه هفته ها پایش بشه و بخش های مختلفش بهینه بشه. 
باید وقت گذاشته بشه که این کار ها رو بشه اتومات کرد. به طور کلی به این فرایند بعدش میگن 
`MLOPs`


## فصل سوم: صفر تا صد یک پروزه یادگیری ماشین برای طبقه‌بندی (classification)


### ساختار مساله طبقه‌بندی 

مبحث طبقه بندی خیلی از موارد قبلی رو داره ولی از نظر مفهومی متفاوت هست با بحث قبلی یعنی رگرسیون. 
تایپ متغیری که اینجا میخوایم پیشبینی کنیم و کلاسش یک چیز گسسته هست در صورتی که توی رگرسیون مثلا خونه یه تایپ پیوسته بود
نحوه ارزیابی هم متفاوته با قبلی. 
برای این بخش از یک دیتاست جدید استفاده میکنیم از دیتاست گل و گیاه ها. 

![[Pasted image 20261115223348.png]]

خب میشه نوع این گل ها رو بر اساس چهار تا فیچر تقسیم و تشخیص داد. اون چهار تا فیچر چیا هستن؟
- طول گل برگ ها های بالا و پایین 
- عرض گل برگ ها بالا و پایین 
حالا میخوایم مدلی بسازیم که بر اساس این فیچر ها کلاس گل ها رو تشخیص بده. 
خب حیلی مشخصه که اگر برای هر کلاس گل یه عدد بذاریم نمیتونه مدل بیاد بگه مثلا یک و نیم. نمیشه گل یک و نیم باشه کلاسش. به طور مشخص باید یا یک باشه یا دو باشه یا سه پس کوانتتیو هست و گستته.
دیتا ست معروفی که میشه استفاده کرد برای طبقه بندی دیتا ست mnist هست که دیتا ست تصویر از عدد ها با دست نوشته های مختلف هست. 

یه خوبی ای که سایکیت لرن داره امکان دانلود دیتاست های معروف هست. برای اینکار باید :

```python
from sklearn.datasets import fetch_openml
mnist_dataset = fetch_openml("mnist_784", as_frame=False)
```

که خب فراخوانیشم سخت نیست. به صورت :

```python
mnist_dataset.data
```

 که خروجی زیر رو میده:
 ```txt
 array([[0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.], ..., [0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.]], shape=(70000, 784))
 ```

که این میشه فیچرهامون. ۷۰۰۰۰ تا سطر(تصویر) که به ازای هر کدوم ۷۸۴ تا ستون تحویل داده میشه.