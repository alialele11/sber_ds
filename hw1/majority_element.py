class Solution:
    def majority_element(self, nums: List[int]) -> int:
        nums = sorted(nums)
        count = 1
        n = int(len(nums) / 2)
        for i in range(len(nums)):
            if nums[i] == nums[i - 1]:
                count += 1
            else:
                count = 1
            if count > n:
                return nums[i]
