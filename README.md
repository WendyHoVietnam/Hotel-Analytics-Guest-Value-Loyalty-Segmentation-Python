# Hotel Analytics: Guest Value Loyalty Segmentation | Python

Author: Ho Nguyen Nhu Quynh

Tool used: Python (pandas, numpy, matplotlib) on Google Colab

## Executive Summary

- **Revenue:** The hotel's guests generated 30.6M in total, 81.6% from lodging and 18.4% from other spending.
- **Spending is spread out:** The top 20% of guests bring only 45.7% of revenue, so the hotel does not depend on a few big spenders.
- **Loyalty is rare:** Only 2.3% of guests stayed more than once. They bring 4.5% of revenue and spend about twice as much per guest (about 965 vs. about 483).
- **Most valuable groups:** "Big spenders to win back" and "Recent big spenders" are 38.5% of guests but 63.6% of revenue. The win-back group last stayed about 708 days ago.
- **Markets:** French, British and German guests average 486 to 516 per guest, while Portuguese guests average 363.
- **Never stayed:** 23.8% of guests in the database never stayed. Direct bookings are over-represented among them, while corporate and group bookings are under-represented.

## Table of Contents
1. Background & Business Question
2. Dataset Description
3. Data Preparation
4. Data Analysis
5. Conclusion

## 1. Background & Business Questions

**Business context:** This project analyzes 83,590 guests of a hotel in Lisbon, Portugal. The analysis is for the hotel's Marketing Manager, Revenue Manager and General Manager.

**Business question:** 

1. What is the hotel's total revenue, and how is it split between lodging and other spending?
2. Which guests spend the most, and what share of revenue do they bring?
3. Which guests are loyal, meaning they stay often and recently?
4. When we combine recency, frequency and monetary value (RFM), what guest segments appear, and how big and valuable is each?

A supporting question (Question 5) looks at the guests who never stayed.

**Approach:**

1. Check and clean the data, and flag invalid values.
2. Split guests into three groups: never stayed, stayed with no revenue, and stayed with revenue (the RFM population).
3. Score the RFM population on Recency, Frequency and Monetary value, and name the segments with rules.
4. For each question: run the code, read the result, explain the insight, and give a recommendation for the right manager.

## 2. Dataset

- **Source:** Kaggle (dataset: nantonio/a-hotels-customers-dataset)
- **Size:** 83,590 guests (one row per guest) and 31 variables
- **Guest:** Nationality, Age
- **Booking:** AverageLeadTime, DaysSinceCreation, DistributionChannel, MarketSegment
- **Stay and spending:** LodgingRevenue, OtherRevenue, BookingsCheckedIn, BookingsCanceled, BookingsNoShowed, PersonsNights, RoomNights, DaysSinceLastStay, DaysSinceFirstStay
- **Special requests:** a group of yes/no columns starting with SR (for example SRKingSizeBed, SRQuietRoom)
- **Hashed identifiers:** NameHash and DocIDHash, not used in the analysis

## 3. Data Preparation

### 3.1 Check the data

**Code**

```python
print(df.columns.tolist())
df.isna().sum()[df.isna().sum()>0]
```

**Result:** There are 3,779 missing values, all in Age.

**Code**

```python
df.describe().T[['count','min','max','mean','50%']]
```

**Result**

| Column | Finding | What to do |
| --- | --- | --- |
| `Age` | Ranges from -11 to 122, plus 3,779 missing | Treat ages below 18 and above 100 as invalid and set them to NaN. Keep missing ages as an "Unknown" age group instead of filling them in. |
| `AverageLeadTime` | Minimum of -1 | Probably a placeholder for "no booking"; set to NaN. |
| `DaysSinceLastStay`, `DaysSinceFirstStay` | Minimum of -1 | Probably "never stayed"; handle together with zero check-ins. |
| `LodgingRevenue`, `OtherRevenue` | Heavily skewed: lodging mean 299 vs. median 234, max 21,781; other mean 68 vs. median 38.5, max 7,730 | Compare means with medians, and score with quantiles (RFM) instead of raw values. |
| `BookingsCheckedIn`, `PersonsNights`, `RoomNights` | Minimum of 0 and some extreme maxes (66 check-ins) | Count guests who never stayed and decide how to treat them. |
| `BookingsCanceled`, `BookingsNoShowed` | Means of 0.002 and 0.0006, so only about 0.2% and 0.06% of guests have any | Too rare to drive segmentation. |
| `SR...` special requests | King bed (35%), twin bed (14%), quiet room (9%) and high floor (5%) are used; most others are under 1% | Not used in the RFM analysis. |

**Code**

```python
print("Age <18:",(df['Age']<18).sum())
print("Age>100:",(df['Age'] >100).sum())
print("Leadtime<0:",(df['AverageLeadTime']<0).sum())
print("DaysSinceLastStay<0:",(df['DaysSinceLastStay']<0).sum())
print("DaysSinceFirstStay<0:",(df['DaysSinceFirstStay']<0).sum())
print("Zero check-ins:",(df['BookingsCheckedIn']==0).sum())
print("Zero lodging revenue:",(df['LodgingRevenue']==0).sum())
print("Duplicate rows (ignoring ID):",df.drop(columns='ID').duplicated().sum())
```

**Result**

```
Age <18: 4414
Age>100: 8
Leadtime<0: 10
DaysSinceLastStay<0: 19920
DaysSinceFirstStay<0: 19920
Zero check-ins: 19920
Zero lodging revenue: 20408
Duplicate rows (ignoring ID): 80
```

<aside>
💡

**Insight**

- The 19,920 guests with zero check-ins are exactly the guests with -1 in both "days since" columns, so -1 means "never stayed".
- Zero total revenue (lodging plus other) is 20,205 guests: the 19,920 who never stayed plus 285 who stayed with no revenue (see the crosstab in 3.2).
</aside>

**Cleaning rules**

| Issue | Rule |
| --- | --- |
| Age below 18 or above 100 | Set to NaN |
| All missing ages | Keep as "Unknown" age group; don't fill with the median |
| Lead time below 0 | Set to NaN |
| Never stayed (19,920) | Flag as `NeverStayed`, analyze separately |
| Stayed, zero revenue (285) | Flag, and exclude from the monetary scoring |
| Duplicates (80) | Check first, then decide |

### 3.2 Clean the data

**Code**

```python
df_clean = df.copy()
df_clean.loc[(df_clean["Age"]<18)|(df_clean["Age"]>100),"Age"] = np.nan
df_clean.loc[(df_clean["AverageLeadTime"]<0),"AverageLeadTime"] = np.nan
df_clean['TotalRevenue'] = df_clean['LodgingRevenue']+df_clean['OtherRevenue']
df_clean['NeverStayed'] = df_clean['BookingsCheckedIn'] ==0

#Do revenue and check-ins line up
print(pd.crosstab(df_clean['NeverStayed'],df_clean['TotalRevenue']==0))

#What do the under-18 ages look like
print(df.loc[df['Age']<18,'Age'].value_counts().sort_index().head(10))

#Are the duplicated mostly never-stayed guests?
dups = df_clean.drop(columns='ID').duplicated(keep=False)
print(df_clean.loc[dups,'NeverStayed'].value_counts())
```

**Result: how the guest base is split**

| Group | Count | Treatment |
| --- | --- | --- |
| Never stayed | 19,920 | Own segment, outside RFM |
| Stayed, zero revenue | 285 | Flagged, excluded from RFM scoring |
| Stayed, with revenue | 63,385 | **RFM population** |
| Invalid ages | set to NaN | "Unknown" age group |
| Duplicates | 144 rows | Of the 144 rows involved in duplicates, 118 are never-stayed guests. Those guests have almost empty records (no revenue, no stays), so identical rows are expected, and they are probably different people, not true duplicates. **Decision: keep them all** |

### 3.3 Build the RFM population and scores

**Code**

```python
#Create the RFM group
rfm = df_clean[(~df_clean['NeverStayed'])&(df_clean['TotalRevenue']>0)].copy()
print(len(rfm))

print(rfm['BookingsCheckedIn'].value_counts().sort_index().head(8))
print(rfm['BookingsCheckedIn'].value_counts(normalize=True).sort_index().head(8))

#Frequency: set by hand, because most guests have 1 stay
rfm['F_score'] = pd.cut(rfm['BookingsCheckedIn'],
                        bins=[0,1,2,np.inf],
                        labels=[1,2,3]).astype(int)
#Recency: fewer days since last stay = better, so labels run from 5 down to 1
rfm['R_score'] = pd.qcut(rfm['DaysSinceLastStay'],5,
                         labels=[5,4,3,2,1]).astype(int)
#Monetary: more revenue = better
rfm['M_score']=pd.qcut(rfm['TotalRevenue'],5,
                       labels=[1,2,3,4,5]).astype(int)
for c in ['R_score','F_score','M_score']:
  print(rfm[c].value_counts().sort_index())
```

<aside>
💡

**Insight**

- 97.7% of the 63,385 guests stayed exactly once (61,955 guests), so Frequency is scored with three hand-set bands (1 stay, 2 stays, 3 or more) instead of five equal groups.
- Recency and Monetary are split into five equal-sized groups of about 12,700 guests each.
</aside>

## 4. Analysis

### Question 1: What is the hotel's total revenue, and how is it split between lodging and other spending?

**Code**

```python
Total_lodging_revenue=df_clean['LodgingRevenue'].sum()
print('Total_lodging_revenue:',Total_lodging_revenue)

Total_other_revenue=df_clean['OtherRevenue'].sum()
print('Total_other_revenue:',Total_other_revenue)

Total_revenue=df_clean['TotalRevenue'].sum()
print('Total_revenue:',Total_revenue)

Pct_lodging_revenue=Total_lodging_revenue*100/Total_revenue
print('%Lodging Revenue:',Pct_lodging_revenue)

Pct_other_revenue=Total_other_revenue*100/Total_revenue
print('%Other Revenue:',Pct_other_revenue)
```

**Result**

```
Total_lodging_revenue: 24976866.46
Total_other_revenue: 5649775.53
Total_revenue: 30626641.99
%Lodging Revenue: 81.55
%Other Revenue: 18.45
```

<aside>
💡

**Insight**

- The hotel's total revenue is about 30.6 million: 25.0 million from lodging and 5.6 million from other spending.
- Lodging brings 81.6% of revenue and other spending (restaurant, spa and similar) 18.4%.
</aside>

<aside>
✅

**Recommendation (Revenue Manager)**

- Action: Grow non-room spending, for example with packages that include restaurant or spa.
- Caveat: The data does not show what guests buy within other spending, so test packages before scaling.
</aside>

### Question 2: Which guests spend the most, and what share of revenue do they bring?

#### 2a. How concentrated is revenue?

**Code**

```python
cutoff = rfm['TotalRevenue'].quantile(0.8)
top20 = rfm[rfm['TotalRevenue']>=cutoff]

print(f'Cutoff revenue: {cutoff:.2f}')
print(f'Guests in top 20%: {len(top20)}')
print(f"Revenue from top 20%: {top20['TotalRevenue'].sum():.2f}")

top20_share = top20['TotalRevenue'].sum()/rfm['TotalRevenue'].sum()*100
print(f'Share of revenue: {top20_share:.1f}%')
```

**Result**

```
Cutoff revenue: 647.50
Guests in top 20%: 12680
Revenue from top 20%: 14011198.16
Share of revenue: 45.7%
```

<aside>
💡

**Insight**

- The top 20% of guests (12,680 guests, spending 647.50 or more) bring 45.7% of revenue, far below the classic 80/20 rule.
- Revenue is spread fairly evenly: the other 80% of guests still contribute 54.3%.
- A strategy that chases only the top spenders would miss more than half of the revenue.
</aside>

#### 2b. Who are the top spenders? Nationality

**Code**

```python
shares = pd.DataFrame({
    'top20_%': top20['Nationality'].value_counts(normalize=True)*100,
    'all_%': rfm['Nationality'].value_counts(normalize=True)*100
})
shares['diff_pts']=shares['top20_%']-shares['all_%']
shares.sort_values('all_%',ascending=False).head(10).round(1)
```

**Result: share of top spenders vs. all guests, ten largest nationalities**

| Nationality | Top 20% share (%) | All guests share (%) | Difference (points) |
| --- | --- | --- | --- |
| FRA | 16.6 | 14.8 | 1.9 |
| PRT | 8.6 | 14.7 | -6.1 |
| DEU | 11.9 | 12.5 | -0.5 |
| GBR | 10.4 | 10.4 | 0.0 |
| ESP | 5.8 | 6.2 | -0.4 |
| ITA | 4.1 | 4.0 | 0.0 |
| USA | 3.2 | 3.8 | -0.6 |
| BEL | 5.0 | 3.7 | 1.3 |
| NLD | 3.6 | 3.2 | 0.4 |
| BRA | 3.5 | 3.2 | 0.3 |

**Code**

```python
five = rfm[rfm['Nationality'].isin(['FRA','PRT','DEU','GBR','ESP'])]
five.groupby('Nationality')['TotalRevenue'].agg(['mean','median','count']).round(1)
```

**Result: average spend per guest, five largest markets**

| Nationality | Mean | Median | Guests |
| --- | --- | --- | --- |
| FRA | 516.3 | 404.0 | 9,379 |
| GBR | 500.1 | 399.5 | 6,567 |
| DEU | 486.3 | 385.0 | 7,894 |
| ESP | 456.2 | 355.8 | 3,936 |
| PRT | 362.8 | 248.3 | 9,319 |

<aside>
💡

**Insight**

- France has the highest average (516.3) and median (404.0) spend, but only narrowly ahead of the UK (500.1) and Germany (486.3).
- France is over-represented among top spenders (16.6% vs. 14.8% of all guests), and Portugal is under-represented (8.6% vs. 14.7%).
- Portugal is the clear outlier: Portuguese guests average 363, about 30% below French guests, even though the two countries are almost the same size (14.7% and 14.8% of guests).
</aside>

<aside>
✅

**Recommendation (Marketing Manager)**

- Action: Prioritize France, the UK and Germany for high-value campaigns, with France first, and test offers that raise spend among Portuguese guests.
- Caveat: The differences between France, the UK and Germany are small, and the data does not show why Portuguese guests spend less, so test any budget shift first.
</aside>

### Question 3: Which guests are loyal (stay often and recently)?

**Code**

```python
#1. Repeat guests = 2 or more stays (F_score of 2 or 3)
repeat = rfm[rfm['F_score']>=2]
print('Repeat guests: ',len(repeat))
print('Share of all RFM guests (%):',round(len(repeat)/len(rfm)*100,1))

#2. The ones who also stayed recently (R_score of 4 or 5)
recent_repeat=repeat[repeat['R_score']>=4]
print('Repeat AND recent:',len(recent_repeat))

#3. How much revenue do repeat guests bring
repeat_rev=repeat['TotalRevenue'].sum()
print('Repeat guests revenue: ',round(repeat_rev,2))
print('Share of all total revenue (%):',round(repeat_rev/rfm['TotalRevenue'].sum()*100,1))
```

**Result**

```
Repeat guests: 1430
Share of all RFM guests (%): 2.3
Repeat AND recent: 942
Repeat guests revenue: 1379601.3
Share of all total revenue (%): 4.5
```

<aside>
💡

**Insight**

- Loyalty is rare: only 1,430 guests (2.3% of the 63,385 guests with revenue) stayed more than once, and 1,130 of them stayed exactly twice. Of these repeat guests, 942 also stayed recently (R score of 4 or 5).
- Repeat guests are worth about twice as much: they bring 1,379,601 in revenue (4.5% of the total) from 2.3% of guests, and spend about 965 on average vs. about 483 for all guests.
</aside>

<aside>
✅

**Recommendation (Marketing and Revenue Managers)**

- Action: The loyal base is very small, so the biggest opportunity is turning first-time guests into repeat guests, for example with a follow-up offer after the first stay.
- Caveat: The data cannot show whether repeat guests spend more because they return, or whether bigger spenders are simply more likely to return, so test any retention program before scaling it.
</aside>

### Question 4: Which RFM segments appear, and how big and valuable is each?

**Segment rules** (checked in order, the first rule that matches wins)

| Order | Segment | Rule | Meaning |
| --- | --- | --- | --- |
| 1 | **Loyal VIPs** | F ≥ 2 and M ≥ 4 | Come back and spend a lot |
| 2 | **Repeat guests** | F ≥ 2 | Come back, with average or lower spend |
| 3 | **Recent big spenders** | M ≥ 4 and R ≥ 4 | One stay, high spend, stayed recently |
| 4 | **Big spenders to win back** | M ≥ 4 (R ≤ 3) | One stay, high spend, a while ago |
| 5 | **Lost low spenders** | M ≤ 2 and R ≤ 2 | Low spend, long ago |
| 6 | **Recent guests** | R ≥ 4 | Stayed recently, with low or medium spend |
| 7 | **Mid-value, cooling off** | Everyone left | Medium value, not recent |

**Code**

```python
conditions = [
    (rfm['F_score']>=2) & (rfm['M_score']>=4),
    (rfm['F_score']>=2),
    (rfm['M_score']>=4) & (rfm['R_score']>=4),
    (rfm['M_score']>=4),
    (rfm['M_score']<=2) & (rfm['R_score']<=2),
    (rfm['R_score']>=4),
]

names=['Loyal VIPs','Repeat guests','Recent big spenders',
       'Big spenders to win back','Lost low spenders','Recent guests']

rfm['Segment']=np.select(conditions,names,default='Mid-value, cooling off')

seg = rfm.groupby('Segment').agg(
    guests = ('ID','count'),
    revenue = ('TotalRevenue','sum'),
    avg_revenue = ('TotalRevenue','mean'),
    avg_days_since_last_stay =('DaysSinceLastStay','mean')
)

seg['guest_share_%']= (seg['guests']/seg['guests'].sum()*100).round(1)
seg['revenue_share_%']=(seg['revenue']/seg['revenue'].sum()*100).round(1)
seg.sort_values('revenue',ascending=False)
```

**Result**

| Segment | Guests | Revenue | Avg revenue | Avg days since last stay | Guest share % | Revenue share % |
| --- | --- | --- | --- | --- | --- | --- |
| Big spenders to win back | 13,751 | 10,829,393 | 787.5 | 707.5 | 21.7 | 35.4 |
| Recent big spenders | 10,617 | 8,650,641 | 814.8 | 209.5 | 16.8 | 28.2 |
| Mid-value, cooling off | 11,703 | 3,703,817 | 316.5 | 656.9 | 18.5 | 12.1 |
| Recent guests | 13,835 | 3,692,772 | 266.9 | 217.6 | 21.8 | 12.1 |
| Lost low spenders | 12,049 | 2,370,418 | 196.7 | 850.2 | 19.0 | 7.7 |
| Loyal VIPs | 957 | 1,249,336 | 1,305.5 | 319.5 | 1.5 | 4.1 |
| Repeat guests | 473 | 130,265 | 275.4 | 442.4 | 0.7 | 0.4 |

**Code: chart of guest share vs. revenue share**

```python
rfm_seg = rfm.groupby('Segment').agg(
    guests=('ID', 'count'),
    revenue=('TotalRevenue', 'sum'))

rfm_seg['guest_share'] = rfm_seg['guests'] / rfm_seg['guests'].sum() * 100
rfm_seg['revenue_share'] = rfm_seg['revenue'] / rfm_seg['revenue'].sum() * 100
rfm_seg = rfm_seg.sort_values('revenue_share', ascending=True)

ax = rfm_seg[['guest_share', 'revenue_share']].plot(
    kind='barh', figsize=(9, 5), color=['#9AA5B1', '#4C78A8'])

ax.set_title('Share of guests vs. share of revenue by RFM segment')
ax.set_xlabel('% of total')
ax.set_ylabel('')
ax.legend(['Share of guests', 'Share of revenue'])

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=3)

plt.tight_layout()
plt.savefig('rfm_segments.png', dpi=200, bbox_inches='tight')
plt.show()
```

**Result: chart**

!rfm_segments.png

<aside>
💡

**Insight**

- Two segments bring far more revenue than their size: "Big spenders to win back" (21.7% of guests, 35.4% of revenue) and "Recent big spenders" (16.8% of guests, 28.2% of revenue). Together they are 38.5% of guests and 63.6% of revenue.
- Three segments bring much less than their size: "Lost low spenders" (19.0% of guests, 7.7% of revenue), "Recent guests" (21.8% vs. 12.1%) and "Mid-value, cooling off" (18.5% vs. 12.1%).
- The hotel's most valuable group is one that stayed once, spent a lot, and has not come back (last stay about 708 days ago).
- "Recent big spenders" last stayed about 209 days ago, so they can still be retained before they drift into the win-back group.
- "Loyal VIPs" have the highest average spend (about 1,305 per guest) but are only 1.5% of guests, so they matter less to total revenue than the two big-spender segments.
- "Recent guests" is the largest segment by a hair (21.8% vs. 21.7%), but its guests are low or medium spenders.
- The three high-spend segments (M score 4 or 5: Loyal VIPs, Recent big spenders and Big spenders to win back) hold about 68% of revenue with about 40% of guests.
</aside>

<aside>
✅

**Recommendation (Marketing Manager)**

- Action: Run a win-back campaign for "Big spenders to win back", and keep "Recent big spenders" from drifting away with an offer while they are still recent.
- Caveat: The RFM rules were set by judgment and the dataset has no booking dates, so test any win-back campaign on a small group first.
</aside>

### Supporting question 5: Who are the guests who never stayed?

19,920 guests (23.8% of the 83,590) have no stays and no revenue. Do they come from different booking channels and market segments than guests who stayed?

**Code**

```python
df_clean['StayedStatus']=np.where(df_clean['NeverStayed'],'NeverStayed','StayedGuest')
status_counts=df_clean['StayedStatus'].value_counts()
status_counts

pd.crosstab(df_clean['DistributionChannel'],df_clean['StayedStatus'],
            normalize='columns').mul(100).round(1)

pd.crosstab(df_clean['MarketSegment'],df_clean['StayedStatus'],
            normalize='columns').mul(100).round(1)
```

**Result: distribution channel (% of each group)**

| Channel | Never stayed | Stayed | Gap (points) |
| --- | --- | --- | --- |
| Travel Agent/Operator | 81.9 | 82.1 | -0.2 |
| Direct | 16.8 | 13.4 | +3.4 |
| Corporate | 1.2 | 3.7 | -2.5 |
| Electronic Distribution | 0.1 | 0.8 | -0.7 |

**Result: market segment (% of each group)**

| Market segment | Never stayed | Stayed | Gap (points) |
| --- | --- | --- | --- |
| Other | 61.9 | 56.1 | +5.8 |
| Direct | 16.4 | 12.9 | +3.5 |
| Travel Agent/Operator | 12.0 | 14.6 | -2.6 |
| Groups | 8.2 | 12.3 | -4.1 |
| Corporate | 0.9 | 3.1 | -2.2 |
| Complementary | 0.5 | 0.6 | -0.1 |
| Aviation | 0.0 | 0.4 | -0.4 |

<aside>
💡

**Insight**

- Travel agents and operators bring about 82% of guests in both groups, so this channel is the core of the business, but it does not explain who never stays.
- Direct bookings tend not to turn into stays, while Corporate and Group bookings do: Direct is over-represented among guests who never stayed (16.8% vs. 13.4%), and Corporate (1.2% vs. 3.7%) and Groups (8.2% vs. 12.3%) are under-represented.
- "Other" is the largest market segment in both groups and is over-represented among never-stayed guests (+5.8 points), but the dataset does not say what it contains, so no conclusion is drawn about it.
</aside>

<aside>
✅

**Recommendation (Marketing and General Managers)**

- Action: Test confirmation reminders for direct bookings, and keep developing corporate and group accounts.
- Caveat: The data shows who never stayed, not why, so test any change first.
</aside>

## 5. Conclusion & Recommendations

| # | Question | Key finding | Recommendation | For |
| --- | --- | --- | --- | --- |
| 1 | How much revenue, and from what? | Total revenue is 30.6M: 81.6% lodging, 18.4% other spending | Grow non-room spending, for example with packages that include restaurant or spa | Revenue Manager |
| 2 | Who spends the most? | The top 20% of guests (12,680) bring only 45.7% of revenue, so revenue is not concentrated. Average spend: France 516, UK 500, Germany 486, Portugal 363. France is over-represented among top spenders (16.6% vs. 14.8%), Portugal under-represented (8.6% vs. 14.7%) | Prioritize France, the UK and Germany for high-value campaigns. Test offers that raise spend among Portuguese guests | Marketing Manager |
| 3 | Who is loyal? | Only 1,430 guests (2.3%) stayed more than once, and 942 of them also stayed recently. They bring 4.5% of revenue and spend about 965 on average vs. about 483 for all guests | Turn first-time guests into repeat guests, for example with a follow-up offer after the first stay | Marketing and Revenue Managers |
| 4 | What RFM segments exist? | Seven segments. "Big spenders to win back" and "Recent big spenders" are 38.5% of guests but 63.6% of revenue | Run a win-back campaign for the first group (last stay about 708 days ago) and keep the second group (about 209 days ago) from drifting away | Marketing Manager |
| 5 | Who never stayed? | 19,920 guests (23.8%) have no stays. Direct bookings are over-represented among them (+3.4 points), corporate and group bookings under-represented | Test confirmation reminders for direct bookings and keep developing corporate and group accounts | Marketing and General Managers |

### Limitations

- The dataset has no booking dates, so it cannot show trends over time.
- The data shows who never stayed, who returns and who spends more, but not why. Findings are patterns, not proven causes.
- The RFM segment rules and score bands were set by judgment, and different thresholds would move some guests between segments.
- The "Other" market segment is unexplained in the dataset.
- 285 guests who stayed but have no revenue were left out of the RFM scoring.
- The data comes from one hotel, so results may not apply to other hotels.

Every recommendation above should be tested on a small group of guests before it is scaled.

### Possible next steps

- Check whether length of stay (RoomNights) explains why Portuguese guests spend less than French guests.
- Compare the age profile of the top 20% of spenders with all guests.