from django.db import models

from account.entity.account import Account
from django.utils import timezone

from orders.entity.orders_status import OrderStatus


class Orders(models.Model):
    id = models.AutoField(primary_key=True)
    account = models.ForeignKey(Account, on_delete=models.CASCADE, related_name='orders')
    status = models.CharField(max_length=10, choices=OrderStatus.choices, default=OrderStatus.PENDING)
    created_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Orders {self.id} by {self.account}"

    class Meta:
        db_table = 'orders'
        app_label = 'orders'
