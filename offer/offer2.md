# 剑指offer 41-76

## 41. 最小的k个数

### a. partition--O\(n\)

```java
class Solution {
    public List<Integer> getLeastNumbers_Solution(int [] input, int k) {
        List<Integer> res=new ArrayList<>();
        //可能存在k大于input的长度
        if(input==null||k>input.length) return res;
        int left=0,right=input.length-1;
        while(left<=right){
            int mid=partition(input,left,right);
            if(mid==k-1) break;
            if(mid>k) right=mid-1;
            else left=mid+1;
        }
        for(int i=0;i<k;i++)
            res.add(input[i]);
        Collections.sort(res);
        return res;
    }
    private int partition(int[] input,int l,int r){
        int temp=input[l];
        while(l<r){
            while(l<r&&input[r]>temp) r--;
            input[l]=input[r];
            while(l<r&&input[l]<=temp) l++;
            input[r]=input[l];
        }
        input[l]=temp;
        return l;
    }
}
```

### b. 大顶堆--O\(nlogk\)

```java
class Solution {
    public List<Integer> getLeastNumbers_Solution(int [] input, int k) {
        LinkedList<Integer> res=new LinkedList<>();
        if(input==null||input.length==0) return res;
        Queue<Integer> q = new PriorityQueue<>(k, Collections.reverseOrder());
        for(int i=0;i<input.length;i++){
            q.add(input[i]);
            if(q.size()>k) 
                q.remove();
        }
        for(int i=0;i<k;i++)
            res.addFirst(q.poll());
        return res;
    }
}
```

## 42. 数据流中的中位数

```java
class Solution {
    Queue<Integer> q2=new PriorityQueue<>()
        ,q1=new PriorityQueue<>(Collections.reverseOrder());
    public void insert(Integer num) {
        q1.offer(num);
        q2.offer(q1.poll());
        if(q2.size()>q1.size())
            q1.offer(q2.poll());
    }

    public Double getMedian() {
        return q1.size()>q2.size()?(double)q1.peek():(q1.peek()+q2.peek())/2.0;
    }
}
```

## 43. 连续子数组的最大和

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int res=Integer.MIN_VALUE,sum=0;
        for(int i=0;i<nums.length;i++){
            sum=Math.max(sum+nums[i],nums[i]);
            res=Math.max(sum,res);
        }
        return res;
    }
}
```

## 44. 从1到n整数中1出现的次数

在每次循环中仅计算当前位为1的数字有多少个。每次将一个数字分成三部分，当前位、该位前边的数字、该位后边的数字。根据当前位和1的关系，可以分为三种情况：

1. 当前位为0 。例如2**0**3，如果要使其十位为1的话，左边部分可以取0和1，而右边可以取0~9，总个数为2\*10=20。
2. 当前位为1 。例如2**1**3，这时如果左边取0或1的话，右边可以取0-9，个数为2\*10=20；而如果左边取2，右边只能取0-3，个数为3+1=4；总个数为20+4=24。
3. 当前位大于1。例如2**2**3，这时左边可以取0-2，右边可以取0-9，总个数为（2+1）\*10=30。

```java
class Solution {
    public int numberOf1Between1AndN_Solution(int n) {
        if(n<=0) return 0;
        int i=1,high=n,low,cur,res=0;
        while(high>0){
            high=n/(i*10);
            int temp=n%(i*10);
            cur=temp/i;
            low=temp%i;
            if(cur<1){
                res+=high*i;
            }else if(cur==1){
                res+=high*i+low+1;
            }else{
                res+=(high+1)*i;
            }
            i*=10;
        }
        return res;
    }
}
```

## 45. 数字序列中某一位的数字

```java
class Solution {
    public int digitAtIndex(int n) {
        if(n<10) return n;
        long s=9,d=1,base=1;
        while(n>d*s){
            n-=d*s;
            d++;
            s*=10;
            base*=10;
        }
        long number=base+(n-1)/d;
        long digit=n%d==0?0:d-n%d;
        while(digit-->0){
            number/=10;
        }
        return (int)number%10;
    }
}
```

## 46. 把数组排成最小的数

```java
class Solution {
    public String printMinNumber(int[] nums) {
        ArrayList<Integer> list=new ArrayList<>();
        for(int i:nums)
            list.add(i);
        Collections.sort(list,new Comparator<Integer>(){
            public int compare(Integer a,Integer b){
                String s1=a+""+b;
                String s2=b+""+a;
                return s1.compareTo(s2);
            }
        });
        if(list.get(0)==0) return "0";//special case
        String res="";
        for(Integer i:list)
            res+=i;
        return res;
    }
}
```

## 47. 把数字翻译成字符串

```java
class Solution {
    public int getTranslationCount(String s) {
        int[] dp=new int[s.length()+1];
        dp[0]=1;
        dp[1]=1;
        for(int i=1;i<s.length();i++){
            dp[i+1]=dp[i];
            if(s.charAt(i-1)=='1'||(s.charAt(i-1)=='2'&&s.charAt(i)<'6'))
                dp[i+1]+=dp[i-1];
        }
        return dp[s.length()];
    }
}
```

## 48. 礼物的最大价值

```java
class Solution {
    public int getMaxValue(int[][] grid) {
        if(grid.length==0||grid[0].length==0) return 0;
        int m=grid.length,n=grid[0].length;
        int[] dp=new int[n];
        for(int i=0;i<m;i++){
            dp[0]+=grid[i][0];
            for(int j=1;j<n;j++){
                dp[j]=Math.max(dp[j-1],dp[j])+grid[i][j];
            }
        }
        return dp[n-1];
    }
}
```

## 49. 最长不含重复字符的子字符串

```java
class Solution {
    public int longestSubstringWithoutDuplication(String s) {
        Map<Character,Integer> mp=new HashMap<>();
        int res=0,h=0;
        for(int i=0;i<s.length();i++){
            char c=s.charAt(i);
            if(mp.containsKey(c)){
                h=Math.max(h,mp.get(c)+1);
            }
            mp.put(c,i);
            res=Math.max(i-h+1,res);
        }
        return res;
    }
}
```

## 50. 丑数

```java
class Solution {
    public int getUglyNumber(int n) {
        int[] res=new int[n];
        res[0]=1;
        int t2=0,t3=0,t5=0;
        for(int i=1;i<n;i++){
            res[i]=Math.min(Math.min(res[t2]*2,res[t3]*3),res[t5]*5);
            if(res[i]==res[t2]*2) t2++;
            if(res[i]==res[t3]*3) t3++;
            if(res[i]==res[t5]*5) t5++;
        }
        return res[n-1];
    }
}
```

## 51. 字符串中第一个只出现一次的字符

```java
class Solution {
    public char firstNotRepeatingChar(String s) {
        int[] mp=new int[256];
        for(int i=0;i<s.length();i++)
            mp[s.charAt(i)]++;
        for(int i=0;i<s.length();i++)
            if(mp[s.charAt(i)]==1)
                return s.charAt(i);
        return '#';
    }
}
```

## 52. 字符流中第一个只出现一次的字符

用map记录字符出现的次数。

用一个队列维护只出现过一次的字符。

```java
class Solution {    
    //Insert one char from stringstream   
    Queue<Character> q=new LinkedList<>();
    int[] mp=new int[256];
    public void insert(char ch){
        mp[ch]++;
        if(mp[ch]>1){
            while(!q.isEmpty()&&mp[q.peek()]>1) q.poll();
        }else{
            q.offer(ch);
        }
    }
    //return the first appearence once char in current stringstream
    public char firstAppearingOnce(){
        return q.isEmpty()?'#':q.peek();
    }
}
```

## 53. 数组中的逆序对

在归并排序的过程中计算逆序对。

```java
class Solution {
    int res=0;
    public int inversePairs(int[] nums) {
        merge(nums,new int[nums.length],0,nums.length-1);
        return res;
    }
    private void merge(int[] nums,int[] temp,int L,int R){
        if(L>=R) return;
        int mid=L+(R-L)/2;
        merge(nums,temp,L,mid);
        merge(nums,temp,mid+1,R);
        int i=L,j=mid+1,index=L;
        while(i<=mid&&j<=R){
            if(nums[i]>nums[j]){
                temp[index++]=nums[j++];
                //nums[j]和前半部分全为逆序
                res+=mid-i+1;
            }else{
                temp[index++]=nums[i++];
            }
        }
        while(i<=mid) temp[index++]=nums[i++];
        while(j<=R) temp[index++]=nums[j++];
        for(i=L;i<=R;i++)
            nums[i]=temp[i];
    }
}
```

## 54. 两个链表的第一个公共结点

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
class Solution {
    public ListNode findFirstCommonNode(ListNode headA, ListNode headB) {
        int len1=0,len2=0;
        ListNode p1=headA,p2=headB;
        while(p1!=null){
            len1++;
            p1=p1.next;
        }
        while(p2!=null){
            len2++;
            p2=p2.next;
        }
        int d=Math.abs(len1-len2);
        if(len1>len2){
            p1=headA;
            p2=headB;
        }else{
            p1=headB;
            p2=headA;
        }
        while(d>0){
            p1=p1.next;
            d--;
        }
        while(p1!=p2){
            p1=p1.next;
            p2=p2.next;
        }
        return p1;
    }
}
```

## 55. 数字在排序数组中出现的次数

二分查找第一个大于等于k的数，和第一个大于等于k+1的数。

```java
class Solution {
    public int getNumberOfK(int[] nums, int k) {
        int l=0,r=nums.length;
        while(l<r){
            int mid=l+(r-l)/2;
            if(nums[mid]>=k) r=mid;
            else l=mid+1;
        }
        //数组中没有k直接返回0
        if(l==nums.length||nums[l]!=k) return 0;
        int temp=l;
        l=temp;
        r=nums.length;
        while(l<r){
            int mid=l+(r-l)/2;
            if(nums[mid]>=k+1) r=mid;
            else l=mid+1;
        }
        return l-temp;
    }
}
```

## 56. 0到n-1中缺失的数字

二分查找。

这道题有很多相似的题目，比如找到0到n-1中唯一重复的数字。

```java
class Solution {
    public int getMissingNumber(int[] nums) {
        int l=0,r=nums.length;
        while(l<r){
            int mid=l+(r-l)/2;
            if(nums[mid]==mid) l=mid+1;
            else r=mid;
        }
        return l;
    }
}
```

## 57. 数组中数值和下标相等的元素

```java
class Solution {
    public int getNumberSameAsIndex(int[] nums) {
        int l=0,r=nums.length-1;
        while(l<=r){
            int mid=l+(r-l)/2;
            if(nums[mid]>mid) r=mid-1;
            else if(nums[mid]<mid) l=mid+1;
            else return mid;
        }
        return -1;
    }
}
```

## 58. 二叉搜索树的第k个结点

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private int ct=0;
    public TreeNode kthNode(TreeNode root, int k) {
        if(root==null) return null;
        TreeNode node=kthNode(root.left,k);
        if(node!=null) return node;
        ct++;
        if(ct==k) return root;
        return kthNode(root.right,k);
    }
}
```

## 59. 二叉树的深度

```java
class Solution {
    public int treeDepth(TreeNode root) {
        if(root==null) return 0;
        return Math.max(treeDepth(root.left),treeDepth(root.right))+1;
    }
}
```

## 60. 平衡二叉树

```java
class Solution {
    boolean flag=true;
    public boolean isBalanced(TreeNode root) {
        depth(root);
        return flag;
    }
    private int depth(TreeNode root){
        //flag为false直接退出
        if(root==null||!flag) return 0;
        int l=depth(root.left),r=depth(root.right);
        if(Math.abs(l-r)>1) flag=false;
        return Math.max(l,r)+1;
    }
}
```

## 61. 数组中只出现一次的两个数字

```java
class Solution {
    public int[] findNumsAppearOnce(int[] nums) {
        int m=0;
        for(int i:nums){
            m^=i;
        }
        int mask=1;
        //将mask设为m中最后一个1出现的位置
        while((mask&m)==0)
            mask<<=1;
        int[] res=new int[2];
        for(int i:nums){
            if((i&mask)!=0){
                res[0]^=i;
            }else{
                res[1]^=i;
            }
        }
        return res;
    }
}
```

## 62. 数组中唯一只出现一次的数字

```java
class Solution {
    public int findNumberAppearingOnce(int[] nums) {
        int[] bitsum=new int[32];
        for(int i=0,n=1;i<32;i++){
            for(int j=0;j<nums.length;j++){
                bitsum[i]+=(n&nums[j])==0?0:1;
            }
            n<<=1;
        }
        int res=0;
        for(int i=31;i>=0;i--){
            res<<=1;
            res+=bitsum[i]%3;
        }
        return res;
    }
}
```

## 63. 和为S的两个数字

```java
class Solution {
    public int[] findNumbersWithSum(int[] nums, int target) {
        Arrays.sort(nums);
        int l=0,r=nums.length-1;
        int[] res=new int[2];
        while(l<r){
            if(nums[l]+nums[r]==target){
                res[0]=nums[l];
                res[1]=nums[r];
                break;
            }else if(nums[l]+nums[r]<target){
                l++;
            }else{
                r--;
            }
        }
        return res;
    }
}
```

## 64. 和为S的连续正数序列

```java
class Solution {
    public List<List<Integer> > findContinuousSequence(int sum) {
       List<List<Integer>> res=new ArrayList<>();
       int l=1,r=2,mid=(1+sum)/2,total=3;
       while(l<mid){
           if(total==sum){
               List<Integer> list=new ArrayList<>();
               for(int i=l;i<=r;i++)
                   list.add(i);
               res.add(list);
               total-=l;
               l++;
           }else if(total<sum){
               r++;
               total+=r;
           }else{
               total-=l;
               l++;
           }
       }
       return res;
    }
}
```

## 65. 翻转单词顺序

```java
class Solution {
    public String reverseWords(String s) {
        String[] ss=s.split(" ");
        String res="";
        for(int i=ss.length-1;i>=0;i--){
            res+=ss[i]+" ";
        }
        return res.substring(0,res.length()-1);
    }
}
```

## 66. 左旋转字符串

```java
class Solution {
    public String leftRotateString(String str,int n) {
        if(str.length()==0||n==0) return str;
        int len=str.length();
        n%=len;
        str+=str;
        return str.substring(n,n+len);
    }
}
```

## 67.  滑动窗口的最大值

```java
class Solution {
    public int[] maxInWindows(int[] nums, int k) {
        int[] res=new int[nums.length-k+1];
        Deque<Integer> q=new LinkedList<>();
        for(int i=0;i<nums.length;i++){
            if(!q.isEmpty()&&q.peekFirst()<i-k+1) q.pollFirst();
            while(!q.isEmpty()&&nums[q.peekLast()]<=nums[i]){
                q.pollLast();
            }
            q.offerLast(i);
            if(i>=k-1){
                res[i-k+1]=nums[q.peek()];
            }
        }
        return res;
    }
}
```

## 68. 骰子的点数

```java
class Solution {
    public int[] numberOfDice(int n) {
        int[][] dp=new int[2][n*6+1];
        int turn=1;
        dp[0][0]=1;
        for(int i=1;i<=n;i++){
            for(int j=i;j<=6*i;j++){
                //每次计算前需要清零
                dp[turn][j]=0;
                //j-k>=i-1是因为前i-1次骰子的和最小为i-1
                for(int k=1;k<=6&&j-k>=i-1;k++){
                    dp[turn][j]+=dp[1-turn][j-k];
                }
            }
            turn=1-turn;
        }
        return turn==1?Arrays.copyOfRange(dp[0],n,dp[0].length)
            :Arrays.copyOfRange(dp[1],n,dp[1].length);
    }
}
```

## 69. 扑克牌的顺子

```c++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        vector<int> mp(14);
        int minval=14,maxval=0;
        for(auto num:nums){
            if(num>0){
                if(mp[num]>0) return false;
                mp[num]++;
                minval=min(minval,num);
                maxval=max(maxval,num);
            }
        }
        return maxval-minval<5;
    }
};
```

## 70. 圆圈中最后剩下的数字

```java
class Solution {
    public int lastRemaining(int n, int m) {
        List<Integer> circle=new LinkedList<>();
        for(int i=0;i<n;i++)
            circle.add(i);
        int index=0;
        while(circle.size()>1){
            index=(m-1+index)%circle.size();
            circle.remove(index);
        }
        return circle.get(0);
    }
}
```

## 71. 股票的最大利润

```java
class Solution {
    public int maxDiff(int[] nums) {
        int res=0,min=Integer.MAX_VALUE;
        for(int i:nums){
            min=Math.min(min,i);
            res=Math.max(res,i-min);
        }
        return res;
    }
}
```

## 72. 求1+2+…+n

```java
class Solution {
    public int getSum(int n) {
        int sum=n;
        boolean t=n>0&&(sum+=getSum(n-1))>0;
        return sum;
    }
}
```

## 73. 不用加减乘除做加法

```java
class Solution {
    public int add(int num1, int num2) {
        int sum=0,carry=0;
        do{
            sum=num1^num2;
            carry=(num1&num2)<<1;
            num1=sum;
            num2=carry;
        }while(carry!=0);
        return sum;
    }
}
```

## 74. 构建乘积数组

```java
class Solution {
    public int[] multiply(int[] A) {
        int len=A.length;
        int[] B=new int[len];
        for(int i=0,p=1;i<len;p*=A[i],i++)
            B[i]=p;
        for(int i=len-1,p=1;i>=0;p*=A[i],i--)
            B[i]*=p;
        return B;
    }
}
```

## 75. 把字符串转换成整数

```java
class Solution {
    public int strToInt(String str) {
        str=str.trim();
        long res=0;
        int flag=1,index=0;
        if(index<str.length()&&(str.charAt(index)=='+'||str.charAt(index)=='-')){
            if(str.charAt(index)=='-') flag=-1;
            index++;
        }
        while(index<str.length()&&str.charAt(index)>='0'&&str.charAt(index)<='9'){
            res=res*10+flag*(str.charAt(index)-'0');
            if(res>0x7FFFFFFF) return 0x7FFFFFFF;
            if(res<0x80000000) return 0x80000000;
            index++;
        }
        return (int)res;
    }
}
```

## 76. 树中两个结点的最低公共祖先

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null||root==p||root==q) return root;
        TreeNode left=lowestCommonAncestor(root.left,p,q);
        TreeNode right=lowestCommonAncestor(root.right,p,q);
        if(left!=null&&right!=null) return root;
        return left!=null?left:right;
    }
}
```

