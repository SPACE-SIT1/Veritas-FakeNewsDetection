import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# อ่านข้อมูล
file_path = "data/FakeNews.xlsx"
dataset = pd.read_excel(file_path)


# ตรวจสอบข้อมูลเบื้องต้น
print("ข้อมูลเบื้องต้น:")
print(dataset.head())
print("\nขนาดของข้อมูล (rows, columns):", dataset.shape)
print("\nข้อมูลที่หายไปในแต่ละคอลัมน์:")
print(dataset.isnull().sum())

# ลบข้อมูลที่ไม่ใช่ข่าวจริง/ข่าวปลอม
dataset = dataset[dataset['Label'].isin([True, False])].copy()

# แปลงคอลัมน์ Label ให้เป็นตัวเลข
labelencoder_target = LabelEncoder()
dataset['Label'] = labelencoder_target.fit_transform(dataset['Label'])
print("\nค่าที่มีในคอลัมน์ 'Label' หลังแปลงเป็นตัวเลข:")
print(dataset['Label'].value_counts())

# แปลง Link_Of_News เป็นฟีเจอร์บ่งชี้แหล่งข่าวที่เชื่อถือได้
dataset['Link_Of_News'] = dataset['Link_Of_News'].apply(lambda x: 1 if 'politifact.com'
                                                        in x else 0)

# ใช้ One-hot encoding สำหรับ Source
source_dummies = pd.get_dummies(dataset['Source'], drop_first=True)
dataset = pd.concat([dataset, source_dummies], axis=1)

# ใช้ TF-IDF เพื่อแปลง News_Headline เป็นฟีเจอร์
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_headline_tfidf = tfidf_vectorizer.fit_transform(dataset['News_Headline'])

# แปลงฟีเจอร์ TF-IDF ให้อยู่ในรูปแบบ DataFrame
X_text_df = pd.DataFrame(
    X_headline_tfidf.toarray(),  # แปลงจาก Sparse Matrix เป็น Dense Matrix
    columns=tfidf_vectorizer.get_feature_names_out()  # ตั้งชื่อคอลัมน์จาก TF-IDF features
)

# ==============================
# Before & After TF-IDF Visualization
# ==============================

# เลือก headline ตัวอย่าง 5 ข่าวแรก
sample_headlines = dataset['News_Headline'].head(5).reset_index(drop=True)

# เลือกเฉพาะคำที่มีค่า TF-IDF มากที่สุดใน 5 ข่าวแรก
sample_tfidf = X_text_df.head(5)

# เอาเฉพาะ top words ที่มีค่า TF-IDF สูงสุด
top_tfidf_words = sample_tfidf.sum(axis=0).sort_values(ascending=False).head(15).index
sample_tfidf_top = sample_tfidf[top_tfidf_words]

# สร้างรูป Before: Raw Headlines
plt.figure(figsize=(12, 4))
plt.axis('off')

text_display = "Before TF-IDF: Raw News Headlines\n\n"
for i, headline in enumerate(sample_headlines, 1):
    text_display += f"{i}. {headline}\n\n"

plt.text(0, 1, text_display, fontsize=10, va='top', wrap=True)
plt.title("Before TF-IDF: Text Data", fontsize=14, fontweight='bold')
plt.show()

# สร้างรูป After: TF-IDF Matrix Heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(sample_tfidf_top, annot=True, cmap='Blues', fmt=".2f")
plt.title("After TF-IDF: Text Converted into Numerical Features", fontsize=14, fontweight='bold')
plt.xlabel("TF-IDF Features / Words")
plt.ylabel("News Headline Index")
plt.show()

# เลือกฟีเจอร์ Non-text (ตัวเลขและ One-hot Encoded)
X_other_df = dataset[['Link_Of_News'] + list(source_dummies.columns)]

# รวมฟีเจอร์ทั้งหมดด้วย pd.concat
X_combined = pd.concat([X_other_df.reset_index(drop=True), 
                        X_text_df.reset_index(drop=True)], axis=1)

# เช็กข้อมูลหายอีกที
print("\nข้อมูลที่หายไปในแต่ละคอลัมน์หลังการแปลง:")
print(X_combined.isnull().sum())

# ปรับข้อมูลให้มีมาตรฐานเดียวกัน
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# แยก Target (Label)
y = dataset['Label']

# แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X_combined_scaled, y, test_size=0.2, random_state=42)

# สร้างโมเดล Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# ประเมินผล
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy ของโมเดล: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# สร้าง Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# แสดงผลลัพธ์
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "True"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ใช้ Cross-Validation
cv_scores = cross_val_score(model, X_combined_scaled, y, cv=5, scoring='accuracy')
print("\nCross-Validation Accuracy:")
print(f"ค่าเฉลี่ยความแม่นยำ: {cv_scores.mean():.4f}")
print(f"ความแม่นยำแต่ละรอบ: {cv_scores}")

# เรียกใช้ความสำคัญของฟีเจอร์จากโมเดล
feature_importances = model.feature_importances_

# แสดงฟีเจอร์ที่สำคัญที่สุด
features = X_combined.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False).head(10)

# สร้างกราฟ
# การกระจาย Label (ข่าวจริง/ปลอม)
plt.figure(figsize=(6, 4))
sns.countplot(data=dataset, x='Label')
plt.title("Distribution of Labels (True vs Fake)")
plt.xlabel("Label (0: Fake, 1: True)")
plt.ylabel("Count")
plt.show()

# การกระจายข้อมูลตามแหล่งข่าว (Source)
top_sources = dataset['Source'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_sources.values, y=top_sources.index, palette='viridis', legend=False)
plt.title("Top 10 Sources of News")
plt.xlabel("Count")
plt.ylabel("Source")
plt.show()

# แยกข่าวจริงและข่าวปลอม
true_headlines = " ".join(dataset[dataset['Label'] == 1]['News_Headline'].dropna())
fake_headlines = " ".join(dataset[dataset['Label'] == 0]['News_Headline'].dropna())


# Top 10 Important Features 
plt.figure(figsize=(10, 6))
plt.barh(importances_df['Feature'], importances_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Top 10 Important Features')
plt.show()