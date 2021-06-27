class Solution:
    def max_sub_array(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        count = 0
        num_of_neg_values = 0
        for i in range(n):
            if nums[i] < 0:
                num_of_neg_values += 1
            if count + nums[i] > 0:
                count += nums[i]
            else:
                count = 0
            ans = max(count, ans)
            if num_of_neg_values == n:
                return max(nums)
        return ans
