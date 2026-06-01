def predict_digits(digit1_logits, digit2_logits):
    digit1_pred = digit1_logits.argmax(dim =1 )
    digit2_pred = digit2_logits.argmax(dim =1 )
    return digit1_pred, digit2_pred

def reason_even_odd(digit1_pred, digit2_pred):
      sum = digit1_pred + digit2_pred
      label = sum % 2
      return label

"""
0 = even
1 = odd
"""
