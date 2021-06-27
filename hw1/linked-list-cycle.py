# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def has_cycle(self, head: ListNode) -> bool:
        fast = head
        slow = head
        if head is None:
            return False
        if head.next is None:
            return False
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
