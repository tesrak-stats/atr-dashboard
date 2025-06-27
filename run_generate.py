def detect_triggers_and_goals(daily, intraday):
    """
    CLEAN LOGIC:
    1. OPEN triggers: Only record if they DON'T complete at OPEN (actionable only)
    2. Intraday triggers: Only record if they didn't already trigger at OPEN (no double-counting)
    """
    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                 -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    
    results = []
    
    for i in range(1, len(daily)):
        try:
            # Use PREVIOUS day's data for level calculation
            previous_row = daily.iloc[i-1]  
            current_row = daily.iloc[i]     
            
            previous_close = previous_row['Close']  
            previous_atr = previous_row['ATR']      
            trading_date = current_row['Date']
            
            # Date filtering
            if hasattr(trading_date, 'strftime'):
                date_str = trading_date.strftime('%Y-%m-%d')
            elif isinstance(trading_date, str):
                date_str = trading_date[:10]
            else:
                date_str = str(trading_date)[:10]
            
            if date_str < '2014-01-02':
                continue
            
            if pd.isna(previous_atr) or pd.isna(previous_close):
                continue
            
            # Generate levels using PREVIOUS day's close + ATR
            level_map = generate_atr_levels(previous_close, previous_atr)
            
            # Get intraday data for trading date
            day_data = intraday[intraday['Date'] == pd.to_datetime(trading_date).date()].copy()
            if day_data.empty:
                continue

            day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
            day_data.reset_index(drop=True, inplace=True)

            # STEP 1: Identify what triggers at OPEN
            open_candle = day_data.iloc[0]
            open_price = open_candle['Open']
            
            open_triggered_up = set()
            open_triggered_down = set()
            
            # Check what triggers at OPEN
            for level in [lvl for lvl in fib_levels if lvl >= 0]:
                if open_price >= level_map[level]:
                    open_triggered_up.add(level)
            
            for level in [lvl for lvl in fib_levels if lvl <= 0]:
                if open_price <= level_map[level]:
                    open_triggered_down.add(level)

            # STEP 2: Process OPEN triggers (only actionable ones)
            for level in open_triggered_up:
                # Check if this OPEN trigger has actionable goals (non-OPEN completions)
                actionable_goals = []
                
                for goal_level in fib_levels:
                    if goal_level == level:  # Skip same level
                        continue
                    
                    if goal_level not in level_map:
                        continue
                        
                    goal_price = level_map[goal_level]
                    goal_hit = False
                    goal_time = ''
                    
                    # Determine if this is continuation or retracement
                    if goal_level > level:
                        goal_type = 'Continuation'
                        
                        # Check if goal completes at OPEN (non-actionable)
                        if open_price >= goal_price:
                            continue  # Skip OPEN completions
                        
                        # Check subsequent candles for upside goal
                        for _, row in day_data.iloc[1:].iterrows():  # Skip OPEN candle
                            if row['High'] >= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    else:
                        goal_type = 'Retracement'
                        
                        # Check if goal completes at OPEN (non-actionable)
                        if open_price <= goal_price:
                            continue  # Skip OPEN completions
                        
                        # Check subsequent candles for downside goal
                        for _, row in day_data.iloc[1:].iterrows():  # Skip OPEN candle
                            if row['Low'] <= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    
                    # This is an actionable goal
                    actionable_goals.append({
                        'GoalLevel': goal_level,
                        'GoalPrice': round(goal_price, 2),
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': goal_time if goal_hit else '',
                        'GoalClassification': goal_type
                    })
                
                # Only record OPEN trigger if it has actionable goals
                if actionable_goals:
                    for goal_info in actionable_goals:
                        results.append({
                            'Date': trading_date,
                            'Direction': 'Upside',
                            'TriggerLevel': level,
                            'TriggerTime': 'OPEN',
                            'TriggerPrice': round(level_map[level], 2),
                            'GoalLevel': goal_info['GoalLevel'],
                            'GoalPrice': goal_info['GoalPrice'],
                            'GoalHit': goal_info['GoalHit'],
                            'GoalTime': goal_info['GoalTime'],
                            'GoalClassification': goal_info['GoalClassification'],
                            'PreviousClose': round(previous_close, 2),
                            'PreviousATR': round(previous_atr, 2),
                            'SameTime': False,  # All OPEN triggers recorded are actionable
                            'RetestedTrigger': 'No'
                        })

            # Process OPEN downside triggers
            for level in open_triggered_down:
                # Check if this OPEN trigger has actionable goals (non-OPEN completions)
                actionable_goals = []
                
                for goal_level in fib_levels:
                    if goal_level == level:  # Skip same level
                        continue
                    
                    if goal_level not in level_map:
                        continue
                        
                    goal_price = level_map[goal_level]
                    goal_hit = False
                    goal_time = ''
                    
                    # Determine if this is continuation or retracement
                    if goal_level < level:
                        goal_type = 'Continuation'
                        
                        # Check if goal completes at OPEN (non-actionable)
                        if open_price <= goal_price:
                            continue  # Skip OPEN completions
                        
                        # Check subsequent candles for downside goal
                        for _, row in day_data.iloc[1:].iterrows():  # Skip OPEN candle
                            if row['Low'] <= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    else:
                        goal_type = 'Retracement'
                        
                        # Check if goal completes at OPEN (non-actionable)
                        if open_price >= goal_price:
                            continue  # Skip OPEN completions
                        
                        # Check subsequent candles for upside goal
                        for _, row in day_data.iloc[1:].iterrows():  # Skip OPEN candle
                            if row['High'] >= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    
                    # This is an actionable goal
                    actionable_goals.append({
                        'GoalLevel': goal_level,
                        'GoalPrice': round(goal_price, 2),
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': goal_time if goal_hit else '',
                        'GoalClassification': goal_type
                    })
                
                # Only record OPEN trigger if it has actionable goals
                if actionable_goals:
                    for goal_info in actionable_goals:
                        results.append({
                            'Date': trading_date,
                            'Direction': 'Downside',
                            'TriggerLevel': level,
                            'TriggerTime': 'OPEN',
                            'TriggerPrice': round(level_map[level], 2),
                            'GoalLevel': goal_info['GoalLevel'],
                            'GoalPrice': goal_info['GoalPrice'],
                            'GoalHit': goal_info['GoalHit'],
                            'GoalTime': goal_info['GoalTime'],
                            'GoalClassification': goal_info['GoalClassification'],
                            'PreviousClose': round(previous_close, 2),
                            'PreviousATR': round(previous_atr, 2),
                            'SameTime': False,  # All OPEN triggers recorded are actionable
                            'RetestedTrigger': 'No'
                        })

            # STEP 3: Process INTRADAY triggers (only if not already triggered at OPEN)
            intraday_triggered_up = {}
            intraday_triggered_down = {}

            # Process each intraday candle (skip OPEN candle)
            for idx, row in day_data.iloc[1:].iterrows():
                high = row['High']
                low = row['Low']
                time_label = row['Time']

                # Check upside triggers (only if not already triggered at OPEN)
                for level in [lvl for lvl in fib_levels if lvl >= 0]:
                    if level in open_triggered_up:  # Skip if already triggered at OPEN
                        continue
                    if level in intraday_triggered_up:  # Skip if already triggered intraday
                        continue
                    
                    trigger_price = level_map[level]
                    if high >= trigger_price:
                        intraday_triggered_up[level] = {
                            'TriggerLevel': level,
                            'TriggerTime': time_label,
                            'TriggeredRow': idx,
                            'TriggerPrice': trigger_price
                        }

                # Check downside triggers (only if not already triggered at OPEN)
                for level in [lvl for lvl in fib_levels if lvl <= 0]:
                    if level in open_triggered_down:  # Skip if already triggered at OPEN
                        continue
                    if level in intraday_triggered_down:  # Skip if already triggered intraday
                        continue
                    
                    trigger_price = level_map[level]
                    if low <= trigger_price:
                        intraday_triggered_down[level] = {
                            'TriggerLevel': level,
                            'TriggerTime': time_label,
                            'TriggeredRow': idx,
                            'TriggerPrice': trigger_price
                        }

            # Process intraday upside triggers and goals
            for level, trigger_info in intraday_triggered_up.items():
                trigger_row = trigger_info['TriggeredRow']
                trigger_candle = day_data.iloc[trigger_row]
                
                for goal_level in fib_levels:
                    if goal_level == level:  # Skip same level
                        continue
                    
                    if goal_level not in level_map:
                        continue
                        
                    goal_price = level_map[goal_level]
                    goal_hit = False
                    goal_time = ''
                    
                    # Determine if this is continuation or retracement
                    if goal_level > level:
                        goal_type = 'Continuation'
                        
                        # Check if goal is hit on same candle as trigger
                        if trigger_candle['High'] >= goal_price:
                            goal_hit = True
                            goal_time = trigger_info['TriggerTime']
                        else:
                            # Check subsequent candles for upside goal
                            for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                                if row['High'] >= goal_price:
                                    goal_hit = True
                                    goal_time = row['Time']
                                    break
                    else:
                        goal_type = 'Retracement'
                        
                        # Check subsequent candles for downside goal
                        for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                            if row['Low'] <= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    
                    results.append({
                        'Date': trading_date,
                        'Direction': 'Upside',
                        'TriggerLevel': level,
                        'TriggerTime': trigger_info['TriggerTime'],
                        'TriggerPrice': round(trigger_info['TriggerPrice'], 2),
                        'GoalLevel': goal_level,
                        'GoalPrice': round(goal_price, 2),
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': goal_time if goal_hit else '',
                        'GoalClassification': goal_type,
                        'PreviousClose': round(previous_close, 2),
                        'PreviousATR': round(previous_atr, 2),
                        'SameTime': False,  # No same-time scenarios in this clean logic
                        'RetestedTrigger': 'No'
                    })

            # Process intraday downside triggers and goals
            for level, trigger_info in intraday_triggered_down.items():
                trigger_row = trigger_info['TriggeredRow']
                trigger_candle = day_data.iloc[trigger_row]
                
                for goal_level in fib_levels:
                    if goal_level == level:  # Skip same level
                        continue
                    
                    if goal_level not in level_map:
                        continue
                        
                    goal_price = level_map[goal_level]
                    goal_hit = False
                    goal_time = ''
                    
                    # Determine if this is continuation or retracement
                    if goal_level < level:
                        goal_type = 'Continuation'
                        
                        # Check if goal is hit on same candle as trigger
                        if trigger_candle['Low'] <= goal_price:
                            goal_hit = True
                            goal_time = trigger_info['TriggerTime']
                        else:
                            # Check subsequent candles for downside goal
                            for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                                if row['Low'] <= goal_price:
                                    goal_hit = True
                                    goal_time = row['Time']
                                    break
                    else:
                        goal_type = 'Retracement'
                        
                        # Check subsequent candles for upside goal
                        for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                            if row['High'] >= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    
                    results.append({
                        'Date': trading_date,
                        'Direction': 'Downside',
                        'TriggerLevel': level,
                        'TriggerTime': trigger_info['TriggerTime'],
                        'TriggerPrice': round(trigger_info['TriggerPrice'], 2),
                        'GoalLevel': goal_level,
                        'GoalPrice': round(goal_price, 2),
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': goal_time if goal_hit else '',
                        'GoalClassification': goal_type,
                        'PreviousClose': round(previous_close, 2),
                        'PreviousATR': round(previous_atr, 2),
                        'SameTime': False,  # No same-time scenarios in this clean logic
                        'RetestedTrigger': 'No'
                    })

        except Exception as e:
            st.write(f"⚠️ Error processing {trading_date}: {str(e)}")
            continue

    return pd.DataFrame(results)
