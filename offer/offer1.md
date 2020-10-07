# 剑指offer(1-40)

## 1. 找出数组中重复的数字

首先遍历数组，如果有不在0~1中的数字，返回-1。

然后遍历数组，如果nums[i]不等于i，则将nums[i]与nums[nums[i]]交换，将其放到正确的位置。如果nums[nums[i]]已经等于nums[i]，说明该数字重复了，返回nums[i]。

时间复杂度：每个数字最多经过两次交换，需要O(n)时间

空间复杂度：O(1)

```java
class Solution {
    public int duplicateInArray(int[] nums) {
        if(nums==null) return -1;
        int len=nums.length;
        for(int i=0;i<len;i++)
            if(nums[i]>=len||nums[i]<0)
                return -1;
        for(int i=0;i<len;i++){
            while(nums[i]!=i){
                if(nums[nums[i]]==nums[i])
                    return nums[i];
                int temp=nums[i];
                nums[i]=nums[temp];
                nums[temp]=temp;
            }
        }        
        return -1;
    }
}
```

## 2. 不修改数组找出重复的数字

如果把数组看作一个链表，数组的值作为下一个结点的位置，那么数组中重复的值则意味着有多个结点指向同一个结点，也就是出现了环。我们可以通过链表判环的方法找到环的入口，也就找到了重复的值。

时间复杂度：O(n)

空间复杂度：O(1)

```java
class Solution {
    public int duplicateInArray(int[] nums) {
        int fast=nums[nums[0]],slow=nums[0];
        while(fast!=slow){
            fast=nums[nums[fast]];
            slow=nums[slow];
        }
        fast=0;
        while(fast!=slow){
            slow=nums[slow];
            fast=nums[fast];
        }
        return slow;
    }
}
```

## 3. 二维数组中的查找

定义两个指针row和col，分别指向第一行和最后一列。

如果当前的值小于target，说明这一行的值都小于target，row++；如果当前值大于target，这一列的值都大于target，col--。

时间复杂度：每步会排除一行或一列，如果行为n列为m的话，时间复杂度为O(n+m)。

```java
class Solution {
    public boolean searchArray(int[][] array, int target) {
        if(array==null||array.length==0||array[0].length==0) return false;
        int row=0,col=array[0].length-1;
        while(row<array.length&&col>=0){
            if(array[row][col]==target)
                return true;
            else if(array[row][col]>target){
                col--;
            }else{
                row++;
            }
        }
        return false;
    }
}
```

## 4. 替换空格

使用双指针算法。

1. 先遍历字符串，计算空格的个数，从而得到新的字符串的长度。
2. 将字符串设置为新的长度。
3. 使用两个指针i,j指向原字符串的末尾和新字符串的末尾，从后往前进行替换。

时间复杂度：O(n)

```java
class Solution {
    public String replaceSpaces(StringBuffer str) {
        int sum=0,len=str.length();
        for(int i=0;i<len;i++){
            if(str.charAt(i)==' ')
                sum++;
        }
        int newLen=sum*2+len;
        str.setLength(newLen);
        for(int i=newLen-1,j=len-1;j>=0;j--){
            if(str.charAt(j)==' '){
                str.replace(i-2,i+1,"%20");
                i-=3;
            }else{
                str.setCharAt(i--,str.charAt(j));
            }
        }
        return str.toString();
    }
}
```

## 5. 从尾到头打印链表

##### a.（遍历链表）O(n)

逆置链表后重新遍历输出。

时间复杂度：O(n)

空间复杂度：O(1)

```java
class Solution {
    public int[] printListReversingly(ListNode head) {
        if(head==null) return new int[0];
        ListNode pre=null,cur=head,temp;
        int len=0;
        while(cur!=null){
            temp=cur.next;
            cur.next=pre;
            pre=cur;
            cur=temp;
            len++;
        }
        int[] res=new int[len];
        int index=0;
        cur=pre;//逆置后pre指向链表头
        while(cur!=null){
            res[index++]=cur.val;
            cur=cur.next;
        }
        return res;
    }
}
```

##### b. (使用头插法)O(n)

时间复杂度：O(n)

空间复杂度：O(n)

```java
class Solution {
    public int[] printListReversingly(ListNode head) {
        if(head==null) return new int[0];
        ListNode cur=head;
        LinkedList<Integer> list=new LinkedList<>();
        while(cur!=null){
            list.addFirst(cur.val);
            cur=cur.next;
        }
        int len=list.size();
        int[] res=new int[len];
        for(int i=0;i<len;i++){
            res[i]=list.get(i);
        }
        return res;
    }
}
```

## 6. 重建二叉树

递归建树。

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        TreeNode root=helper(preorder,inorder,0,0,inorder.length-1);
        return root;
    }
    private TreeNode helper(int[] preorder,int[] inorder,int preL,int inL,int inR){
        if(inL>inR) return null;
        TreeNode root=new TreeNode(preorder[preL]);
        int i=inL;
        for(;inorder[i]!=root.val;i++);
        int len=i-inL;
        root.left=helper(preorder,inorder,preL+1,inL,i-1);
        root.right=helper(preorder,inorder,preL+len+1,i+1,inR);
        return root;
    }
}
```

## 7. 二叉树的下一个节点

在二叉树的中序遍历中，二叉树的下一个节点有两种情况：

1. 该节点有右子树。这时候下一个节点是右子树中最左边的节点。
2. 该节点没有右子树。这时候判断节点与其父节点的关系：如果该节点是父节点的左节点，那么父节点就是下一个节点；如果是右节点，那么继续对父节点进行判断，直到父节点为null为止。因为二叉树的节点中有指向父节点的指针，所以可以很容易地实现。

```java
class Solution {
    public TreeNode inorderSuccessor(TreeNode p) {
        if(p.right!=null){
            TreeNode node=p.right;
            while(node.left!=null)
                node=node.left;
            return node;
        }
        TreeNode fa=p.father;
        while(fa!=null&&fa.right==p){
            p=fa;
            fa=fa.father;
        }
        return fa;
    }
}
```

## 8. 用两个栈实现队列

```java
class MyQueue {
    
    Stack<Integer> s1,s2;
    /** Initialize your data structure here. */
    public MyQueue() {
        s1=new Stack<Integer>();
        s2=new Stack<Integer>();
    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
        s1.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if(s2.isEmpty()){
            while(!s1.isEmpty()){
                s2.push(s1.pop());
            }
        }
        return s2.pop();
    }
    
    /** Get the front element. */
    public int peek() {
        if(s2.isEmpty()){
            while(!s1.isEmpty())
                s2.push(s1.pop());
        }
        return s2.peek();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
        return s1.isEmpty()&&s2.isEmpty();
    }
}
```

## 9. 斐波那契数列

时间复杂度：O(n)

空间复杂度：O(1)

``` java
class Solution {
    public int Fibonacci(int n) {
         int a=0,b=1;
         for(int i=0;i<n;i++){
             b=a+b;
             a=b-a;
         }
         return a;
    }
}
```

## 10. 旋转数组的最小数字

使用二分查找时，需要满足一个条件：有一种性质能够区分前半部分和后半部分。

比如，当数组中没有重复数字时，数组的前半部分nums[i]>=nums[0]，而后半部分都小于nums[0]，这样我们可以通过二分查找找到第一个小于nums[0]的数字。

但是当数组中有重复的数字时，由于后半部分中的数字也可能满足nums[i]==nums[0]，所以二分查找可能会失败。这时候应该对数组进行处理，删除数组末尾等于nums[0]的数字。

时间复杂度：删除末位数字时最坏复杂度为O(n)，二分查找O(log n)，所以总的时间复杂度为O(n)。

```java
class Solution {
    public int findMin(int[] nums) {
        if(nums==null||nums.length==0) return -1;
        int len=nums.length-1;
        while(len>=0&&nums[0]==nums[len]) len--;
        int left=0,right=len+1;
        while(left<right){
            int mid=left+((right-left)>>1);
            if(nums[mid]<nums[0])
                right=mid;
            else
                left=mid+1;
        }
        return left<len+1?nums[left]:nums[0];
    }
}
```

## 11. 矩阵中的路径

回溯法，用visit数组标记走过的路径。

```java
class Solution {
    public boolean hasPath(char[][] matrix, String str) {
        if(matrix==null||matrix.length==0||matrix[0].length==0) return false;
        int row=matrix.length,col=matrix[0].length;
        for(int i=0;i<row;i++)
            for(int j=0;j<col;j++)
                if(dfs(matrix,str,new boolean[row][col],i,j,0))
                    return true;
        return false;
    }
    private boolean dfs(char[][] matrix,String str,boolean[][] visit,int i,int j,int index){
        if(i<0||i>=matrix.length||j<0||j>=matrix[0].length) return false;
        if(visit[i][j]||matrix[i][j]!=str.charAt(index)) return false;
        index++;
        visit[i][j]=true;
        if(index==str.length()) return true;
        boolean res=dfs(matrix,str,visit,i,j+1,index)||dfs(matrix,str,visit,i+1,j,index)
        ||dfs(matrix,str,visit,i-1,j,index)||dfs(matrix,str,visit,i,j-1,index);
        visit[i][j]=false;
        return res;
    }
}
```

## 12. 机器人的运动范围

可以用bfs也可以用dfs。

```java
class Solution {
    public int movingCount(int threshold, int rows, int cols)
    {
        if(rows==0||cols==0) return 0;
        return dfs(threshold,rows,cols,0,0,new boolean[rows][cols]);
        
    }
    private int dfs(int t,int rows,int cols,int i,int j,boolean[][] visited){
        if(i<0||i>=rows||j<0||j>=cols||visited[i][j]||getSum(i,j)>t) return 0;
        visited[i][j]=true;
        return dfs(t,rows,cols,i,j+1,visited)+dfs(t,rows,cols,i+1,j,visited)+
        dfs(t,rows,cols,i,j-1,visited)+dfs(t,rows,cols,i-1,j,visited)+1;//不要忘记+1
    }
    private int getSum(int row,int col){
        int res=0;
        while(row>0){
            res+=row%10;
            row/=10;
        }
        while(col>0){
            res+=col%10;
            col/=10;
        }
        return res;
    }
}
```

## 13. 剪绳子

假设n=k1+k2+……+ki，求k1\*k2\*……\*ki的值。

首先ki=1时对乘积没有改变，所以ki!=1。

ki=2或3时，不应该拆分。

ki=4和5时，分解成2+2和2+3乘积最大。

ki>5时，我们可以将其拆成3+(ki-3)或2+(ki-2)，得到的乘积3(ki-3)>2(ki-2)>ki，所以应该将其拆分，并且结果中有尽可能多的3。此时要考虑一个特例，当ki%3==1时，3\*1的乘积要小于2\*2，要进行特殊处理。

```java
class Solution {
    public int maxProductAfterCutting(int length)
    {
        if(length<4) return length-1;
        int a=length/3,b=length%3;
        if(b==1){
           a--;
           b=4;
        }
        if(b==0) b=1;
        return b*(int)Math.pow(3,a);
    }
}
```

## 14. 二进制中1的个数

每次做n&(n-1)操作，都会把n右边的1置为0。这样循环1的个数次之后，就可以得到结果。

```java
class Solution {
    public int NumberOf1(int n)
    {
        int ct=0;
        while(n!=0){
            ct++;
            n=n&(n-1);
        }
        return ct;
    }
}
```

## 15. 数值的整数次方

采用乘方公式。

时间复杂度：O(log n)

```java
class Solution {
    public double Power(double base, int exponent) {
        if(base==0) return 0.0;
        if(exponent==0) return 1.0;
        if(exponent==1) return base;
        if(exponent<0) return 1.0/Power(base,-exponent);//指数可能为负
        double res=Power(base,exponent>>1);//位运算效率高
        res*=res;
        if((exponent&1)==1){//==的优先级高于&
            res*=base;
        }
        return res;
  }
}
```

## 16. 在O(1)时间删除链表结点

将当前节点的值改为下一个节点的值，然后删除下一个节点。

```java
class Solution {
    public void deleteNode(ListNode node) {
        node.val=node.next.val;
        node.next=node.next.next;
    }
}
```

## 17. 删除链表中重复的节点

1. 设置一个头节点dummy方便处理。
2. pre指向上一个未重复的节点，cur遍历链表，如果一个节点的值没有重复，就将pre.next设置为该节点。
3. 注意最后将pre.next设置为null。

```java
class Solution {
    public ListNode deleteDuplication(ListNode head) {
        ListNode dummy=new ListNode(0);
        ListNode pre=dummy,cur=head;
        while(cur!=null){
            ListNode temp=cur;
            while(cur!=null&&cur.val==temp.val)
                cur=cur.next;
            if(cur==temp.next){
                pre.next=temp;
                pre=temp;
            }
        }
        pre.next=null;//处理最后一个数重复的情况
        return dummy.next;
    }
}
```

## 18. 正则表达式匹配

主要难点在于对\*号的匹配。

##### a. 递归解法：

```java
class Solution {
    public boolean isMatch(String s, String p) {
        if(s==null||p==null) return false;
        return helper(s.toCharArray(),p.toCharArray(),0,0);
    }
    private boolean helper(char[] s,char[] p,int i,int j){
        if(i==s.length&&j==p.length) return true;
        if(j==p.length) return false;
        if(i==s.length){
            if(j+1<p.length&&p[j+1]=='*')
                return helper(s,p,i,j+2);
            return false;
        }
        if(j+1<p.length&&p[j+1]=='*'){
            if(s[i]==p[j]||p[j]=='.')//匹配时有三种可能
                return helper(s,p,i,j+2)||helper(s,p,i+1,j+2)||helper(s,p,i+1,j);
            return helper(s,p,i,j+2);//不匹配，直接跳过
        }
        if(s[i]==p[j]||p[j]=='.')
            return helper(s,p,i+1,j+1);
        return false;
    }
}
```

##### b. 非递归解法

```java
class Solution {
    public boolean isMatch(String s, String p) {
        if(s==null||p==null) return false;
        char[] sa=s.toCharArray(),pa=p.toCharArray();
        int slen=sa.length,plen=pa.length;
        boolean[][] dp=new boolean[slen+1][plen+1];
        dp[0][0]=true;
        for(int i=0;i<plen;i++){
            if(pa[i]=='*')
                dp[0][i+1]=true;
        }
        for(int i=0;i<slen;i++){
            for(int j=0;j<plen;j++){
                if(pa[j]=='*'){
                    if(pa[j-1]=='.'||pa[j-1]==sa[i])
                        dp[i+1][j+1]=dp[i+1][j-1]||dp[i][j+1]||dp[i+1][j];
                    else
                        dp[i+1][j+1]=dp[i+1][j-1];
                }else if(pa[j]=='.'||pa[j]==sa[i]){
                    dp[i+1][j+1]=dp[i][j];
                }
            }
        }
        return dp[slen][plen];
    }
}
```

## 19. 表示数值的字符串

一个有效数值的格式应该是(1. 有符号数)(.)(2. 无符号数 )(e或E)(3. 有符号数)。

'.'两边的数有一个就行，'e或E'两边的数必须同时存在。

```java
class Solution {
    public boolean isNumber(String s) {
        int[] index=new int[1];
        char[] ss=s.toCharArray();
        int len=ss.length;
        boolean res=isInteger(ss,index);
        if(index[0]<len&&ss[index[0]]=='.'){
            index[0]++;
            //第一个为真时就不会再判断第二个条件，所以应该把函数放在前边
            res=isUnsignedInteger(ss,index)||res;
        }
        if(index[0]<len&&(ss[index[0]]=='E'||ss[index[0]]=='e')){
            index[0]++;
            res=isInteger(ss,index)&&res;
        }
        return index[0]==len&&res;
    }
    private boolean isInteger(char[] s,int[] index){
        if(index[0]<s.length&&(s[index[0]]=='+'||s[index[0]]=='-'))
            index[0]++;
        return isUnsignedInteger(s,index);
    }
    private boolean isUnsignedInteger(char[] s,int[] index){
        int temp=index[0];
        while(index[0]<s.length&&s[index[0]]<='9'&&s[index[0]]>='0')
            index[0]++;
        return index[0]>temp;
    }
}
```

## 20. 调整数组顺序使奇数位于偶数前面

#####  a. 无特殊要求--双指针O(n)

指针i遍历数组，指针j指向i后的第一个奇数，当array[i]为偶数时，交换array[i]和array[j]。

```java
class Solution {
    public void reOrderArray(int [] array) {
        if(array==null) return;
        int len=array.length;
        for(int i=0,j=0;i<len;i++){
            if((array[i]&1)==0){
                if(j<=i) j=i+1;
                while(j<len&&(array[j]&1)==0) j++;
                if(j==len) break;//j遍历完之后直接退出
                int temp=array[i];
                array[i]=array[j];
                array[j]=temp;
            }
        }
    }
}
```

##### b. 交换后奇数与奇数、偶数与偶数之间顺序不变--插入法O(n^2)

过程类似于插入排序，最坏情况下的时间复杂度O(n^2)。

```java
class Solution {
    public void reOrderArray(int [] array) {
        if(array==null) return;
        int len=array.length;
        for(int i=0,k=0;i<len;i++){
            if((array[i]&1)==1){
                int temp=array[i];
                //倒序遍历赋值
                for(int j=i-1;j>=k;j--)
                    array[j+1]=array[j];
                array[k++]=temp;
            }
        }
    }
}
```

## 21. 链表中倒数第k个节点

定义两个指针fast和slow，fast先走k步，然后两指针同步前进，当fast为null时slow指向的便是倒数第k个节点。

```java
class Solution {
    public ListNode findKthToTail(ListNode pListHead, int k) {
        if(pListHead==null||k==0) return null;
        int len=0;
        ListNode slow=pListHead,fast=pListHead;
        while(fast!=null&&k>0){
            fast=fast.next;
            k--;
        }
        if(k>0) return null;
        while(fast!=null){
            fast=fast.next;
            slow=slow.next;
        }
        return slow;
    }
}
```

## 22. 链表中环的入口结点

```java
class Solution {
    public ListNode entryNodeOfLoop(ListNode head) {
        if(head==null) return null;
        ListNode slow=head,fast=head;
        while(fast!=null&&fast.next!=null){//判断fast.next前要判断fast
            fast=fast.next.next;
            slow=slow.next;
            if(fast==slow) break;
        }
        if(fast==null||fast.next==null) return null;
        fast=head;
        while(fast!=slow){
            fast=fast.next;
            slow=slow.next;
        }
        return fast;
    }
}
```

## 23. 反转链表

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head==null) return null;
        ListNode cur=head,pre=null;
        while(cur!=null){
            ListNode temp=cur.next;
            cur.next=pre;
            pre=cur;
            cur=temp;
        }
        return pre;
    }
}
```

## 24. 合并两个排序的链表

```java
class Solution {
    public ListNode merge(ListNode l1, ListNode l2) {
        ListNode dummy=new ListNode(0);
        ListNode node=dummy;
        while(l1!=null&&l2!=null){
            if(l1.val>l2.val){
                node.next=l2;
                node=l2;
                l2=l2.next;
            }else{
                node.next=l1;
                node=l1;
                l1=l1.next;
            }
        }
        if(l1!=null) node.next=l1;
        if(l2!=null) node.next=l2;
        return dummy.next;
    }
}
```

## 25. 树的子结构

```java
class Solution {
    public boolean hasSubtree(TreeNode pRoot1, TreeNode pRoot2) {
        if(pRoot1==null||pRoot2==null) return false;
        if(sametree(pRoot1,pRoot2))return true;
        return hasSubtree(pRoot1.left,pRoot2)||hasSubtree(pRoot1.right,pRoot2);
    }
    private boolean sametree(TreeNode pRoot1, TreeNode pRoot2){
        if(pRoot2==null) return true;
        if(pRoot1==null) return false;
        return pRoot1.val==pRoot2.val&&sametree(pRoot1.left,pRoot2.left)
            &&sametree(pRoot1.right,pRoot2.right);
    }
}
```

## 26. 二叉树的镜像

```java
class Solution {
    public void mirror(TreeNode root) {
        if(root==null) return;
        mirror(root.left);
        mirror(root.right);
        TreeNode temp=root.left;
        root.left=root.right;
        root.right=temp;
    }
}
```

## 27. 对称的二叉树

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return helper(root,root);
    }
    private boolean helper(TreeNode r1,TreeNode r2){
        if(r1==null&&r2==null) return true;
        if(r1==null||r2==null) return false;
        return r1.val==r2.val&&helper(r1.left,r2.right)&&helper(r1.right,r2.left);
    }
}
```

## 28. 顺时针打印矩阵

用四个变量up,down,left,right标记待遍历的矩阵的边界，在一个大循环中进行四次处理。

```java
class Solution {
    public int[] printMatrix(int[][] matrix) {
        if(matrix==null||matrix.length==0||matrix[0].length==0) return new int[0];
        int rows=matrix.length,cols=matrix[0].length;
        int[] res=new int[rows*cols];
        int index=0,left=0,right=cols-1,up=0,down=rows-1;
        while(left<=right&&up<=down){
            for(int i=left;i<=right;i++)
                res[index++]=matrix[up][i];
            if(++up>down) break;
            for(int i=up;i<=down;i++)
                res[index++]=matrix[i][right];
            if(--right<left) break;
            for(int i=right;i>=left;i--)
                res[index++]=matrix[down][i];
            if(--down<up) break;
            for(int i=down;i>=up;i--)
                res[index++]=matrix[i][left];
            left++;
        }
        return res;
    }
}
```

## 29. 包含min函数的栈

```java
class MinStack {

    Stack<Integer> s1,s2;
    /** initialize your data structure here. */
    public MinStack() {
        s1=new Stack<Integer>();
        s2=new Stack<Integer>();
        s2.push(Integer.MAX_VALUE);//避免s2.push()时的判断栈空
    }
    
    public void push(int x) {
        s1.push(x);
        s2.push(Math.min(s2.peek(),x));
    }
    
    public void pop() {
        s1.pop();
        s2.pop();
    }
    
    public int top() {
        return s1.peek();
    }
    
    public int getMin() {
        return s2.peek();
    }
}
```

## 30. 栈的压入、弹出序列

```java
class Solution {
    public boolean isPopOrder(int [] pushV,int [] popV) {
        if(pushV==null||popV==null) return false;
        int len1=pushV.length,len2=popV.length;
        if(len1!=len2) return false;
        Stack<Integer> s=new Stack<>();
        for(int i=0,j=0;i<len1;i++){
            s.push(pushV[i]);
            while(!s.isEmpty()&&s.peek()==popV[j]){
                s.pop();
                j++;
            }
        }
        return s.isEmpty();
    }
}
```

## 31. 不分行从上往下打印二叉树

```java
class Solution {
    public List<Integer> printFromTopToBottom(TreeNode root) {
        List<Integer> res=new ArrayList<Integer>();
        if(root==null) return res;
        Queue<TreeNode> q=new LinkedList<>();
        q.offer(root);
        while(!q.isEmpty()){
            TreeNode node=q.poll();
            res.add(node.val);
            if(node.left!=null) q.offer(node.left);
            if(node.right!=null) q.offer(node.right);
        }
        return res;
    }
}
```

## 32. 分行从上往下打印二叉树

```java
class Solution {
    public List<List<Integer>> printFromTopToBottom(TreeNode root) {
        List<List<Integer>> res=new ArrayList<>();
        if(root==null) return res;
        Queue<TreeNode> q=new LinkedList<>();
        q.offer(root);
        while(!q.isEmpty()){
            int n=q.size();
            List<Integer> list=new ArrayList<>();
            for(int i=0;i<n;i++){
                TreeNode node=q.poll();
                list.add(node.val);
                if(node.left!=null) q.offer(node.left);
                if(node.right!=null) q.offer(node.right);
            }
            res.add(list);
        }
        return res;
    }
}
```

## 33. 之字形打印二叉树

```java
class Solution {
    public List<List<Integer>> printFromTopToBottom(TreeNode root) {
        List<List<Integer>> res=new ArrayList<>();
        if(root==null) return res;
        Queue<TreeNode> q=new LinkedList<>();
        boolean flag=false;
        q.offer(root);
        while(!q.isEmpty()){
            int n=q.size();
            List<Integer> list=new ArrayList<>();
            for(int i=0;i<n;i++){
                TreeNode node=q.poll();
                list.add(node.val);
                if(node.left!=null) q.offer(node.left);
                if(node.right!=null) q.offer(node.right);
            }
            if(flag){
                Collections.reverse(list);
                flag=false;
            }else{
                flag=true;
            }
            res.add(list);
        }
        return res;
    }
}
```

## 34. 二叉搜索树的后序遍历序列

```java
class Solution {
    public boolean verifySequenceOfBST(int [] sequence) {
        if(sequence==null) return false;
        return helper(sequence,0,sequence.length-1);
    }
    private boolean helper(int[] s,int left,int right){
        if(left>=right) return true;
        int root=s[right],k=left;
        for(;s[k]<root;k++);
        for(int i=k;i<right;i++)
            if(s[i]<root)
                return false;
        return helper(s,left,k-1)&&helper(s,k,right-1);
    }
}
```

## 35. 二叉树中和为某一值的路径

```java
class Solution {
    public List<List<Integer>> findPath(TreeNode root, int sum) {
        List<List<Integer>> res=new ArrayList<>();
        dfs(res,new ArrayList<Integer>(),root,sum,0);
        return res;
    }
    private void dfs(List<List<Integer>> res,List<Integer> list,TreeNode node,int sum,int n){
        if(node==null) return;
        n+=node.val;
        list.add(node.val);
        if(node.left==null&&node.right==null){
            if(sum==n)
                res.add(new ArrayList<Integer>(list));
            list.remove(list.size()-1);
            return;
        }
        dfs(res,list,node.left,sum,n);
        dfs(res,list,node.right,sum,n);
        list.remove(list.size()-1);
    }
}
```

## 36. 复杂链表的复刻

1. 每个节点后面复制一个新的节点。
2. 新节点的random指向原节点的random.next。
3. 将新节点和原节点分开。

```java
class Solution {
    public ListNode copyRandomList(ListNode head) {
        if(head==null) return null;
        ListNode node=head;
        while(node!=null){
            ListNode newNode=new ListNode(node.val);
            newNode.next=node.next;
            node.next=newNode;
            node=newNode.next;
        }
        ListNode pre=head;
        while(pre!=null){
            node=pre.next;
            if(pre.random!=null)
                node.random=pre.random.next;
            pre=node.next;
        }
        ListNode newHead=new ListNode(0);
        node=newHead;
        pre=head;
        while(pre!=null){
            node.next=pre.next;
            node=node.next;
            pre.next=node.next;
            pre=node.next;
        }
        return newHead.next;
    }
}
```

## 37. 二叉搜索树与双向链表

```java
public class Solution {
    private TreeNode lastNode=null;
    public TreeNode convert(TreeNode pRootOfTree) {
        if(pRootOfTree==null) return null;
        TreeNode left=Convert(pRootOfTree.left);
        if(lastNode!=null){
            lastNode.right=pRootOfTree;
            pRootOfTree.left=lastNode;
        }
        lastNode=pRootOfTree;
        Convert(pRootOfTree.right);
        return left==null?pRootOfTree:left;
    }
}
```

## 38. 序列化二叉树

```java
class Solution {
    private int index;
    // Encodes a tree to a single string.
    String serialize(TreeNode root) {
        if(root==null) return "#";
        return String.valueOf(root.val)+","+serialize(root.left)+","+serialize(root.right);
    }

    // Decodes your encoded data to tree.
    TreeNode deserialize(String data) {
        if(data==null) return null;
        index=0;
        String[] strs=data.split(",");
        return helper(strs);
    }
    TreeNode helper(String[] strs){
        if(strs[index].equals("#")){
            index++;//不要在if语句中更新
            return null;
        }
        TreeNode root=new TreeNode(Integer.valueOf(strs[index++]));
        root.left=helper(strs);
        root.right=helper(strs);
        return root;
    }
}
```

## 39. 数字排列

在结果数组中，只要保持重复数字的相对位置不变，就可以保证不存在重复的排列方式。

```java
class Solution {
    public List<List<Integer>> permutation(int[] nums) {
        LinkedList<List<Integer>> res=new LinkedList<>();
        if(nums==null||nums.length==0) return res;
        Arrays.sort(nums);
        backtrick(res,new ArrayList<Integer>(),nums,new boolean[nums.length]);
        return res;
    }
    private void backtrick(List<List<Integer>> res,List<Integer> list,int[] nums,boolean[] used){
        if(list.size()==nums.length){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        for(int i=0;i<nums.length;i++){
            if(used[i]||i>0&&nums[i]==nums[i-1]&&!used[i-1]) continue;
            list.add(nums[i]);
            used[i]=true;
            backtrick(res,list,nums,used);
            list.remove(list.size()-1);
            used[i]=false;
        }
    }
}
```

## 40. 数组中出现次数超过一半的数字

```java
class Solution {
    public int moreThanHalfNum_Solution(int[] array) {
        if(array==null) return 0;
        int num=0,times=0;
        for(int i=0;i<array.length;i++){
            if(times==0){
                num=array[i];
                times++;
            }
            else if(num==array[i])
                times++;
            else
                times--;
        }
        return num;
    }
}
```