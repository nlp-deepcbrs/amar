do
   local function getTermLength()
      if sys.uname() == 'windows' then return 80 end
      local tputf = io.popen('tput cols', 'r')
      local w = tonumber(tputf:read('*a'))
      local rc = {tputf:close()}
      if rc[3] == 0 then return w
      else return 80 end 
   end

   local barDone = true
   local previous = -1
   local tm = ''
   local timer
   local times
   local indices
   local termLength = math.min(getTermLength(), 120)
   function progress(current, goal, cost)
      -- defaults:
      local barLength = termLength - 64
      local smoothing = 100 
      local maxfps = 10
      
      -- Compute percentage
      local percent = math.floor(((current) * barLength) / goal)

      -- start new bar
      if (barDone and ((previous == -1) or (percent < previous))) then
         barDone = false
         previous = -1
         tm = ''
         timer = torch.Timer()
         times = {timer:time().real}
         indices = {current}
      else
         io.write('\r')
      end

      --if (percent ~= previous and not barDone) then
      if (not barDone) then
         previous = percent
         -- print bar
         io.write(' [')
         for i=1,barLength do
            if (i < percent) then io.write('=')
            elseif (i == percent) then io.write('>')
            else io.write('.') end
         end
         io.write('] ')
         for i=1,termLength-barLength-4 do io.write(' ') end
         for i=1,termLength-barLength-4 do io.write('\b') end
         -- time stats
         local elapsed = timer:time().real
         local step = (elapsed-times[1]) / (current-indices[1])
         if current==indices[1] then step = 0 end
         local remaining = math.max(0,(goal - current)*step)
         table.insert(indices, current)
         table.insert(times, elapsed)
         if #indices > smoothing then
            indices = table.splice(indices)
            times = table.splice(times)
         end
         -- Print remaining time when running or total time when done.
         if (percent < barLength) then
            tm = ' ETA: ' .. formatTime(remaining)
         else
            tm = ' Tot: ' .. formatTime(elapsed)
         end
         tm = tm .. ' | Step: ' .. formatTime(step)
         io.write(tm)
         io.write(' | Cost: ' .. cost)
         -- go back to center of bar, and print progress
         for i=1,5+#tm+barLength/2 do io.write('\b') end
         io.write(' ', current, '/', goal, ' ')
         -- reset for next bar
         if (percent == barLength) then
            barDone = true
            io.write('\n')
         end
         -- flush
         io.write('\r')
         io.flush()
      end
   end
end

function formatTime(seconds)
   -- decompose:
   local floor = math.floor
   local days = floor(seconds / 3600/24)
   seconds = seconds - days*3600*24
   local hours = floor(seconds / 3600)
   seconds = seconds - hours*3600
   local minutes = floor(seconds / 60)
   seconds = seconds - minutes*60
   local secondsf = floor(seconds)
   seconds = seconds - secondsf
   local millis = floor(seconds*1000)

   -- string
   local f = ''
   local i = 1
   if days > 0 then f = f .. days .. 'D' i=i+1 end
   if hours > 0 and i <= 2 then f = f .. hours .. 'h' i=i+1 end
   if minutes > 0 and i <= 2 then f = f .. minutes .. 'm' i=i+1 end
   if secondsf > 0 and i <= 2 then f = f .. secondsf .. 's' i=i+1 end
   if millis > 0 and i <= 2 then f = f .. millis .. 'ms' i=i+1 end
   if f == '' then f = '0ms' end

   -- return formatted time
   return f
end
